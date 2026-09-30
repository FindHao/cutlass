# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

"""
Unit tests for the compilation listeners of ``BaseDSL``.

External tracers register these listeners to observe every compiled
specialization. A compilation listener runs after each compilation or
in-memory cache hit and receives the module hash, which identifies the
specialization.

The tests compile and launch a small kernel, so they need a CUDA GPU and skip
without one.
"""

import contextvars
import functools
import re
import unittest
from unittest import mock

import torch

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.cache_helpers import JitCacheDict
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import CompileOptions
from cutlass.base_dsl.jit_executor import JitFunctionArtifacts
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import CuTeDSL


@cute.kernel
def _fill(t: cute.Tensor, value: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    t[tidx] = value


@cute.jit
def _launch(t: cute.Tensor, value: cutlass.Int32, threads: cutlass.Constexpr[int]):
    _fill(t, value).launch(grid=[1, 1, 1], block=[threads, 1, 1])


@cute.kernel
def _empty():
    pass


@cute.jit
def _launch_empty(threads: cutlass.Constexpr[int]):
    _empty().launch(grid=[1, 1, 1], block=[threads, 1, 1])


class _Recorder:
    """A listener that records the keyword arguments of every call."""

    def __init__(self):
        self.owners = []
        self.calls = []

    def __call__(self, owner, **kwargs):
        self.owners.append(owner)
        self.calls.append(kwargs)

    def field(self, name):
        return [call[name] for call in self.calls]


def _failing_listener(owner, **kwargs):
    raise ValueError("listener bug")


def _flat_message(exc):
    """Diagnostics are wrapped and colorized before they reach the user, so
    match against the message with that formatting flattened out."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).split())


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class _ListenerTestCase(unittest.TestCase):
    def setUp(self):
        self.dsl = CuTeDSL._get_dsl()
        # A fresh in-memory JIT cache, so whether a call hits the cache does not
        # depend on the tests that ran before.
        self._patch(self.dsl, "jit_cache", JitCacheDict())
        # Caching on, whatever CUTE_DSL_NO_CACHE or CUTE_DSL_KEEP=ptx/cubin/sass
        # say, so the cache-hit tests do not depend on the environment.
        self.patch_envar(
            no_cache=False, keep_ptx=False, keep_cubin=False, keep_sass=False
        )
        self.data = torch.zeros(32, dtype=torch.int32, device="cuda")
        self.t = from_dlpack(self.data)

    def _patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def patch_envar(self, **settings):
        for name, value in settings.items():
            self._patch(self.dsl.envar, name, value)

    def listener_apis(self):
        """``(scope, register, remove)`` for each kind of listener. Every
        ``_launch`` call notifies a compilation listener once, on the
        compilation or cache hit."""
        return {
            "compilation": (
                self.dsl.compilation_listeners,
                self.dsl.register_compilation_listener,
                self.dsl.remove_compilation_listener,
            ),
        }


class TestListenerRegistration(_ListenerTestCase):
    def test_rejects_none_and_non_callables(self):
        for kind, (scope, register, _) in self.listener_apis().items():
            for listener in (None, 42):
                with self.subTest(kind=kind, listener=listener):
                    with self.assertRaises(DSLRuntimeError):
                        register(listener)
            for listeners in (None, 42, [_Recorder(), None], [42]):
                with self.subTest(kind=kind, scoped=listeners):
                    with self.assertRaises(DSLRuntimeError):
                        with scope(listeners):
                            pass

    def test_registering_twice_notifies_once_and_removal_stops_it(self):
        for kind, (_, register, remove) in self.listener_apis().items():
            with self.subTest(kind=kind):
                listener = _Recorder()
                register(listener)
                register(listener)
                try:
                    _launch(self.t, 1, 32)
                finally:
                    remove(listener)
                _launch(self.t, 1, 32)
                self.assertEqual(len(listener.calls), 1)
                # Removing a listener that is not registered is a no-op.
                remove(listener)

    def test_scoped_listeners_apply_inside_the_block_only(self):
        for kind, (scope, _, _) in self.listener_apis().items():
            with self.subTest(kind=kind):
                outer, inner = _Recorder(), _Recorder()
                with scope(outer):
                    with scope([inner, outer]):
                        _launch(self.t, 1, 32)
                    _launch(self.t, 1, 32)
                _launch(self.t, 1, 32)
                # outer is in both scopes, and still runs once per call.
                self.assertEqual(len(outer.calls), 2)
                self.assertEqual(len(inner.calls), 1)

    def test_registered_and_scoped_listener_runs_once(self):
        for kind, (scope, register, remove) in self.listener_apis().items():
            with self.subTest(kind=kind):
                listener = _Recorder()
                register(listener)
                self.addCleanup(remove, listener)
                with scope(listener):
                    _launch(self.t, 1, 32)
                self.assertEqual(len(listener.calls), 1)

    def test_scoped_listeners_follow_the_context(self):
        """Scoped listeners live in a ``ContextVar``: a fresh context does not
        see them, a copied one inherits them, and leaving the block restores
        the previous state, also when the block raises."""
        call = functools.partial(_launch, self.t, 1, 32)
        for kind, (scope, register, remove) in self.listener_apis().items():
            with self.subTest(kind=kind):
                scoped, registered = _Recorder(), _Recorder()
                register(registered)
                self.addCleanup(remove, registered)
                with scope(scoped):
                    call()
                    contextvars.Context().run(call)
                    contextvars.copy_context().run(call)
                self.assertEqual(len(scoped.calls), 2)
                self.assertEqual(len(registered.calls), 3)
                with self.assertRaises(ValueError), scope(scoped):
                    raise ValueError("block failed")
                call()
                self.assertEqual(len(scoped.calls), 2)
                self.assertEqual(len(registered.calls), 4)


class TestCompilationListener(_ListenerTestCase):
    def test_reports_a_compilation_then_a_cache_hit(self):
        listener = _Recorder()
        with self.dsl.compilation_listeners(listener):
            _launch(self.t, 1, 32)
            _launch(self.t, 1, 32)
        self.assertEqual(listener.field("cache_hit"), [False, True])
        first, second = listener.calls
        self.assertIsInstance(first["module_hash"], str)
        self.assertEqual(second["module_hash"], first["module_hash"])
        self.assertEqual(second["function_name"], first["function_name"])

    def test_payload(self):
        listener = _Recorder()
        with self.dsl.compilation_listeners(listener):
            _launch(self.t, 1, 32)
        self.assertEqual(listener.owners, [self.dsl])
        (call,) = listener.calls
        self.assertIsInstance(call["artifacts"], JitFunctionArtifacts)
        self.assertIsInstance(call["compile_options"], CompileOptions)
        self.assertEqual(call["func_body"].__name__, "_launch")
        self.assertTrue(any("_fill" in name for name in call["kernel_info"]))

    def test_each_specialization_has_its_own_hash(self):
        listener = _Recorder()
        with self.dsl.compilation_listeners(listener):
            _launch(self.t, 1, 32)
            _launch(self.t, 1, 16)
        self.assertEqual(listener.field("cache_hit"), [False, False])
        first, second = listener.field("module_hash")
        self.assertNotEqual(first, second)

    def test_hash_is_computed_for_listeners_when_caching_is_off(self):
        """Without caching there is no cache key to compute, but listeners
        still need the hash to tell specializations apart. Caching stays off."""
        self.patch_envar(no_cache=True)
        listener = _Recorder()
        with self.dsl.compilation_listeners(listener):
            _launch(self.t, 1, 32)
            _launch(self.t, 1, 32)
        self.assertEqual(listener.field("cache_hit"), [False, False])
        first, second = listener.field("module_hash")
        self.assertIsInstance(first, str)
        self.assertEqual(first, second)

    def test_no_hash_without_listeners(self):
        """``cute.compile`` never caches, so it hashes the module only when a
        listener needs the hash."""
        self.assertIsNone(cute.compile(_launch, self.t, 1, 32).module_hash)
        for kind, (scope, _, _) in self.listener_apis().items():
            with self.subTest(kind=kind), scope(_Recorder()):
                compiled = cute.compile(_launch, self.t, 1, 32)
                self.assertIsInstance(compiled.module_hash, str)

    def test_listener_may_compile_another_function(self):
        """The listener runs after the compilation has finished, so it can
        compile another function."""
        listener = _Recorder()
        inner = []

        def compile_once(owner, **kwargs):
            listener(owner, **kwargs)
            # Only the outer compilation compiles; the inner one is reported
            # to this listener too.
            if len(listener.calls) == 1:
                inner.append(cute.compile(_launch_empty, 16))

        with self.dsl.compilation_listeners(compile_once):
            compiled = cute.compile(_launch, self.t, 7, 32)
        outer_call, inner_call = listener.calls
        self.assertNotEqual(outer_call["function_name"], inner_call["function_name"])
        self.assertNotEqual(outer_call["module_hash"], inner_call["module_hash"])
        self.assertIsNot(outer_call["kernel_info"], inner_call["kernel_info"])
        self.assertTrue(any("_fill" in name for name in outer_call["kernel_info"]))
        self.assertTrue(any("_empty" in name for name in inner_call["kernel_info"]))
        compiled(self.t, 7)
        inner[0]()
        torch.cuda.synchronize()
        self.assertTrue(torch.all(self.data == 7))

    def test_listener_error_is_wrapped(self):
        with self.dsl.compilation_listeners(_failing_listener):
            with self.assertRaises(DSLRuntimeError) as ctx:
                _launch(self.t, 1, 32)
        self.assertIn(
            "Compilation listener failed: _failing_listener",
            _flat_message(ctx.exception),
        )
        self.assertIsInstance(ctx.exception.__cause__, ValueError)
        # The DSL stays usable.
        _launch(self.t, 2, 32)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(self.data == 2))


if __name__ == "__main__":
    unittest.main()
