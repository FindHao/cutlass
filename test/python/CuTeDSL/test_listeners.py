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
Unit tests for the compilation and launch listeners of ``BaseDSL``.

External tracers register these listeners to observe every compiled
specialization and every call that runs it. A compilation listener runs after
each compilation or in-memory cache hit; a launch listener runs before each call
to a compiled host function, once per call however many kernels it launches,
and receives the runtime arguments of the call. Both receive the module hash,
which is how a tracer ties a call to the compilation behind it.

The tests compile and launch a small kernel, so they need a CUDA GPU and skip
without one.
"""

import contextvars
import functools
import importlib.util
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
def _add(t: cute.Tensor, value: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    t[tidx] = t[tidx] + value


@cute.jit
def _fill_then_add(t: cute.Tensor, value: cutlass.Int32):
    _fill(t, value).launch(grid=[1, 1, 1], block=[32, 1, 1])
    _add(t, value).launch(grid=[1, 1, 1], block=[32, 1, 1])


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


def _bind(call):
    """Name each runtime argument of a launch the way a tracer does: ``args[i]``
    binds to ``arg_names[i]``, and ``kwargs`` are keyed by name already."""
    bound = dict(zip(call["arg_names"], call["args"]))
    bound.update(call["kwargs"])
    return bound


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
        ``_launch`` call notifies each kind once: a compilation listener on the
        compilation or cache hit, and a launch listener on the launch."""
        return {
            "compilation": (
                self.dsl.compilation_listeners,
                self.dsl.register_compilation_listener,
                self.dsl.remove_compilation_listener,
            ),
            "launch": (
                self.dsl.launch_listeners,
                self.dsl.register_launch_listener,
                self.dsl.remove_launch_listener,
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


class TestLaunchListener(_ListenerTestCase):
    def test_arguments_bind_to_their_names_on_every_path(self):
        """Implicit calls and direct calls to a compiled function report the
        runtime arguments, without the Constexpr ``threads``."""
        compiled = cute.compile(_launch, self.t, 1, 32)
        listener = _Recorder()
        with self.dsl.launch_listeners(listener):
            _launch(self.t, 3, 32)
            _launch(self.t, value=4, threads=32)
            compiled(self.t, 5)
            compiled(self.t, value=6)
        self.assertEqual(
            [_bind(call) for call in listener.calls],
            [{"t": self.t, "value": value} for value in (3, 4, 5, 6)],
        )
        for call in listener.calls:
            self.assertEqual(call["arg_names"], ("t", "value"))
        # The listener sees the objects the caller passed.
        self.assertIs(listener.calls[0]["args"][0], self.t)
        # An implicit launch reports the runtime parameters in declaration
        # order, while a direct call reports keyword arguments as keywords.
        self.assertEqual(listener.calls[1]["args"], (self.t, 4))
        self.assertEqual(listener.calls[3]["kwargs"], {"value": 6})
        # The kernel still ran, with the last value.
        torch.cuda.synchronize()
        self.assertTrue(torch.all(self.data == 6))

    def test_launch_carries_the_hash_of_its_compilation(self):
        compilations, launches = _Recorder(), _Recorder()
        with (
            self.dsl.compilation_listeners(compilations),
            self.dsl.launch_listeners(launches),
        ):
            _launch(self.t, 1, 32)
            _launch(self.t, 1, 16)
            compiled = cute.compile(_launch, self.t, 1, 8)
            compiled(self.t, 1)
        self.assertEqual(
            launches.field("module_hash"), compilations.field("module_hash")
        )
        self.assertEqual(
            launches.field("function_name"), compilations.field("function_name")
        )
        self.assertEqual(launches.owners, [self.dsl] * 3)

    def test_compilation_is_reported_before_its_launch(self):
        events = []
        with (
            self.dsl.compilation_listeners(
                lambda owner, **kwargs: events.append("compilation")
            ),
            self.dsl.launch_listeners(lambda owner, **kwargs: events.append("launch")),
        ):
            _launch(self.t, 1, 32)
            _launch(self.t, 1, 32)
        self.assertEqual(events, ["compilation", "launch"] * 2)

    def test_one_notification_per_host_call(self):
        """A launch listener reports each call to a compiled host function once,
        however many kernels the call launches."""
        compiled = cute.compile(_fill_then_add, self.t, 7)
        listener = _Recorder()
        with self.dsl.launch_listeners(listener):
            _fill_then_add(self.t, 7)
            compiled(self.t, 7)
        self.assertEqual(len(listener.calls), 2)
        for call in listener.calls:
            # kernel_info describes the kernels of the compilation.
            self.assertEqual(len(call["kernel_info"]), 2)
        # Both kernels ran: fill with 7, then add 7.
        torch.cuda.synchronize()
        self.assertTrue(torch.all(self.data == 14))

    def test_listener_error_aborts_the_launch(self):
        compiled = cute.compile(_launch, self.t, 1, 32)
        with self.dsl.launch_listeners(_failing_listener):
            with self.assertRaises(DSLRuntimeError) as ctx:
                compiled(self.t, 9)
        self.assertIn(
            "Launch listener failed: _failing_listener", _flat_message(ctx.exception)
        )
        self.assertIsInstance(ctx.exception.__cause__, ValueError)
        # Launch listeners run before the kernel, so it never ran.
        torch.cuda.synchronize()
        self.assertTrue(torch.all(self.data == 0))


@unittest.skipUnless(importlib.util.find_spec("tvm_ffi"), "needs apache-tvm-ffi")
class TestTVMFFILaunch(_ListenerTestCase):
    """With TVM FFI, a compiled function launches through its own ``__call__``
    instead of ``JitCompiledFunction.__call__``."""

    def setUp(self):
        super().setUp()
        self.patch_envar(enable_tvm_ffi=True)
        self.t = from_dlpack(self.data, enable_tvm_ffi=True)

    def test_each_launch_is_reported_once(self):
        compiled = cute.compile(_launch, self.t, 1, 32)
        listener = _Recorder()
        with self.dsl.launch_listeners(listener):
            _launch(self.t, 3, 32)
            compiled(self.t, 4)
            compiled(self.t, value=5)
        self.assertEqual(
            [_bind(call) for call in listener.calls],
            [{"t": self.t, "value": value} for value in (3, 4, 5)],
        )
        # tvm-ffi converts the arguments of a direct call itself, so there are
        # no packed arguments to report.
        self.assertIsNone(listener.calls[1]["exe_args"])
        torch.cuda.synchronize()
        self.assertTrue(torch.all(self.data == 5))

    def test_function_without_runtime_arguments(self):
        """Such a function compiles to the positional-only TVM FFI class,
        which has a ``__call__`` of its own."""
        # Imported here: the module needs tvm_ffi.
        from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunction

        compiled = cute.compile(_launch_empty, 32)
        self.assertIsInstance(compiled, TVMFFIJitCompiledFunction)
        listener = _Recorder()
        with self.dsl.launch_listeners(listener):
            compiled()
        (call,) = listener.calls
        self.assertEqual((call["args"], call["kwargs"]), ((), {}))

    def test_uninitialized_function_is_not_reported(self):
        """A function that cannot run (for example one compiled for another
        architecture) fails before the listeners hear of the call."""
        from cutlass.cutlass_dsl.tvm_ffi_provider import (
            TVMFFIJitCompiledFunction,
            TVMFFIJitCompiledFunctionWithKwargs,
        )

        positional = cute.compile(_launch_empty, 32)
        with_kwargs = cute.compile(_launch, self.t, 1, 32)
        self.assertIsInstance(positional, TVMFFIJitCompiledFunction)
        self.assertIsInstance(with_kwargs, TVMFFIJitCompiledFunctionWithKwargs)
        cases = (
            (
                positional,
                (),
                mock.patch.object(
                    TVMFFIJitCompiledFunction, "__chandle__", return_value=0
                ),
            ),
            (
                with_kwargs,
                (self.t, 1),
                mock.patch.object(with_kwargs, "_kwargs_wrapper", None),
            ),
        )
        for compiled, args, uninitialized in cases:
            with self.subTest(cls=type(compiled).__name__):
                listener = _Recorder()
                with uninitialized, self.dsl.launch_listeners(listener):
                    with self.assertRaises(DSLRuntimeError):
                        compiled(*args)
                self.assertEqual(listener.calls, [])


if __name__ == "__main__":
    unittest.main()
