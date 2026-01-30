# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import importlib
import inspect
from typing import get_type_hints


def public_symbols(mod) -> list[str]:
    """Return the list of public names for *mod* (honours ``__all__``)."""

    if hasattr(mod, "__all__") and isinstance(mod.__all__, list | tuple):
        return list(mod.__all__)

    def is_public(name: str) -> bool:
        if name.startswith("_"):
            return False
        return not inspect.ismodule(getattr(mod, name))

    return sorted(filter(is_public, dir(mod)))


def format_type(t) -> str:
    """Format a type annotation to a readable string."""
    if t is inspect.Parameter.empty:
        return ""
    if hasattr(t, "__name__"):
        return t.__name__
    # Handle generic types like list[str], Optional[int], etc.
    s = str(t)
    # Clean up common patterns
    for prefix in ("typing.", "warp.types.", "warp.", "newton._src.", "newton."):
        s = s.replace(prefix, "")
    return s


def get_signature(func, name: str) -> str:
    """Get a formatted signature string for a function/method."""
    try:
        sig = inspect.signature(func)
        params = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            annotation = format_type(param.annotation)
            if param.default is not inspect.Parameter.empty:
                # Has default value
                if annotation:
                    params.append(f"{pname}: {annotation} = ...")
                else:
                    params.append(f"{pname}=...")
            else:
                if annotation:
                    params.append(f"{pname}: {annotation}")
                else:
                    params.append(pname)

        ret = format_type(sig.return_annotation)
        ret_str = f" -> {ret}" if ret else ""
        return f"{name}({', '.join(params)}){ret_str}"
    except (ValueError, TypeError):
        return f"{name}(...)"


def get_init_instance_attrs(cls) -> set[str]:
    """Extract instance attribute names set in __init__ by parsing source."""
    import textwrap

    attrs = set()
    try:
        source = inspect.getsource(cls.__init__)
    except (TypeError, OSError):
        return attrs

    # Dedent source to handle indented class methods
    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return attrs

    for node in ast.walk(tree):
        # Look for self.xxx = ... assignments
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    if isinstance(target.value, ast.Name) and target.value.id == "self":
                        if not target.attr.startswith("_"):
                            attrs.add(target.attr)
        # Also handle annotated assignments: self.xxx: type = ...
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Attribute):
                if isinstance(node.target.value, ast.Name) and node.target.value.id == "self":
                    if not node.target.attr.startswith("_"):
                        attrs.add(node.target.attr)

    return attrs


def get_class_members(cls, include_signatures: bool = False):
    """Return dict of public methods and properties for a class."""
    methods = []
    properties = []
    instance_attrs = []
    init_sig = None

    # Get __init__ signature
    if include_signatures and hasattr(cls, "__init__"):
        try:
            init_sig = get_signature(cls.__init__, "__init__")
        except Exception:
            pass

    # Get instance attributes from __init__
    init_attrs = get_init_instance_attrs(cls)

    for name in dir(cls):
        if name.startswith("_"):
            continue

        try:
            attr = getattr(cls, name)
        except AttributeError:
            continue

        # Check if it's defined in this class or a parent we care about
        # Skip inherited builtins
        if hasattr(attr, "__objclass__") and attr.__objclass__ in (object, type):
            continue

        if isinstance(attr, property):
            properties.append(name)
        elif callable(attr) or isinstance(attr, (classmethod, staticmethod)):
            if include_signatures:
                methods.append(get_signature(attr, name))
            else:
                methods.append(name)

    # Add instance attributes from __init__ that aren't already captured as properties
    for attr_name in init_attrs:
        if attr_name not in properties:
            instance_attrs.append(attr_name)

    return {
        "methods": sorted(methods),
        "properties": sorted(properties),
        "instance_attrs": sorted(instance_attrs),
        "init_sig": init_sig,
    }


def get_function_info(func, name: str, include_signatures: bool = False) -> dict:
    """Get info about a function."""
    if include_signatures:
        return {"type": "function", "name": name, "signature": get_signature(func, name)}
    return name


def get_symbols(mod_name: str, include_members: bool = False, include_signatures: bool = False):
    module = importlib.import_module(mod_name)
    all_symbols = public_symbols(module)

    children = []
    for name in all_symbols:
        attr = getattr(module, name)
        if inspect.ismodule(attr):
            children.append(get_symbols(f"{mod_name}.{name}", include_members, include_signatures))
        elif inspect.isclass(attr):
            if include_members or include_signatures:
                members = get_class_members(attr, include_signatures)
                children.append({"type": "class", "name": name, "members": members})
            else:
                children.append(name)
        elif inspect.isfunction(attr):
            if include_signatures:
                children.append(get_function_info(attr, name, include_signatures))
            else:
                children.append(name)
        else:
            children.append(name)

    return (mod_name.split(".")[-1], children)


def print_symbols(sym_dict, indent=0, show_signatures=False):
    name, children = sym_dict[0], sym_dict[1]
    print(f"{'   ' * indent}{name}:")

    symbols = []
    functions = []
    classes = []
    submodules = []
    for child in children:
        if isinstance(child, str):
            symbols.append(child)
        elif isinstance(child, dict) and child.get("type") == "class":
            classes.append(child)
        elif isinstance(child, dict) and child.get("type") == "function":
            functions.append(child)
        elif isinstance(child, tuple):
            submodules.append(child)
        else:
            symbols.append(str(child))

    # sort
    symbols.sort()
    functions.sort(key=lambda x: x["name"])
    classes.sort(key=lambda x: x["name"])
    submodules.sort(key=lambda x: x[0])

    for sym in symbols:
        print(f"{'   ' * (indent + 1)}{sym}")

    for func in functions:
        if show_signatures:
            print(f"{'   ' * (indent + 1)}{func['signature']}")
        else:
            print(f"{'   ' * (indent + 1)}{func['name']}()")

    for cls in classes:
        cls_name = cls["name"]
        members = cls["members"]
        methods = members["methods"]
        properties = members["properties"]
        instance_attrs = members.get("instance_attrs", [])
        init_sig = members.get("init_sig")

        if methods or properties or instance_attrs or init_sig:
            if show_signatures and init_sig:
                print(f"{'   ' * (indent + 1)}{cls_name}({init_sig.split('(', 1)[1] if '(' in init_sig else '...'}")
            else:
                print(f"{'   ' * (indent + 1)}{cls_name}:")
            for prop in properties:
                print(f"{'   ' * (indent + 2)}.{prop}")
            for attr in instance_attrs:
                print(f"{'   ' * (indent + 2)}.{attr}  [instance]")
            for method in methods:
                if show_signatures and "(" in method:
                    print(f"{'   ' * (indent + 2)}.{method}")
                else:
                    print(f"{'   ' * (indent + 2)}.{method}()")
        else:
            print(f"{'   ' * (indent + 1)}{cls_name}")

    print()
    for sub in submodules:
        print_symbols(sub, indent=indent + 1, show_signatures=show_signatures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print public API symbols")
    parser.add_argument(
        "--members",
        action="store_true",
        help="Include class methods and properties",
    )
    parser.add_argument(
        "--signatures",
        action="store_true",
        help="Include function/method signatures with parameter types",
    )
    args = parser.parse_args()

    include_members = args.members or args.signatures
    print_symbols(
        get_symbols("newton", include_members=include_members, include_signatures=args.signatures),
        show_signatures=args.signatures,
    )
