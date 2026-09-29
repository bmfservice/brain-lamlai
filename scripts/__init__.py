"""Operator-only maintenance commands.

This package marker is load-bearing: the ``gguf`` dependency pulled in by vLLM
installs its own top-level ``scripts`` package into site-packages, and a regular
package shadows a namespace package regardless of ``sys.path`` order.
"""
