import ast
from langchain_text_splitters.base import Language, TextSplitter
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC
import logging

logger = logging.getLogger(__name__)


class Chunk(ABC):
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size


class FunctionChunk(Chunk):
    """Container for a function chunk with comprehensive metadata."""

    def __init__(
        self,
        function_name: str,
        function_signature: str,
        start_line: int,
        end_line: int,
        original_chunk_text: str,
        docstring: Optional[str] = None,
        decorators: Optional[List[str]] = None,
        ast_node_type: str = "function",
        complexity_score: int = 1,
    ):
        self.function_name = function_name
        self.function_signature = function_signature
        self.start_line = start_line
        self.end_line = end_line
        self.original_chunk_text = original_chunk_text
        self.docstring = docstring
        self.decorators = decorators or []
        self.ast_node_type = ast_node_type
        self.complexity_score = complexity_score
        self.file_extension = Path(file_path).suffix
        self.chunk_type = "function"  # Identifier for vector store

    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format."""
        return {
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "function_signature": self.function_signature,
            "line_numbers": {"start": self.start_line, "end": self.end_line},
            "function_name": self.function_name,
            "original_chunk_text": self.original_chunk_text,
            "chunk_type": self.chunk_type,
            "metadata": {
                "file_extension": self.file_extension,
                "ast_node_type": self.ast_node_type,
                "complexity_score": self.complexity_score,
                "docstring": self.docstring,
                "decorators": self.decorators,
            },
        }

    def get_enhanced_embedding_text(self) -> str:
        """Generate enhanced text for embeddings using AST metadata.

        Combines function code with semantic context:
        - Decorators (e.g., @staticmethod, @property)
        - Function signature with type hints
        - Docstring (if present)
        - Complexity hint
        - Original code

        This gives the embedding model more semantic context than raw code alone.
        """
        parts = []

        # Add decorators as semantic hints
        if self.decorators:
            parts.append("Decorators: " + ", ".join(self.decorators))

        # Add function signature with full type information
        parts.append(f"Signature: {self.function_signature}")

        # Add docstring for semantic understanding
        if self.docstring:
            parts.append(f"Documentation: {self.docstring}")

        # Add complexity as a hint for algorithm understanding
        if self.complexity_score > 1:
            parts.append(f"Complexity: {self.complexity_score}")

        # Add the actual code
        parts.append(f"Code:\n{self.original_chunk_text}")

        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"FunctionChunk({self.function_name} @ {self.relative_path}:{self.start_line}-{self.end_line})"


class ClassChunk(Chunk):
    """Container for a class chunk with comprehensive metadata."""

    def __init__(
        self,
        class_name: str,
        start_line: int,
        end_line: int,
        original_chunk_text: str,
        docstring: Optional[str] = None,
        base_classes: Optional[List[str]] = None,
        decorators: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        complexity_score: int = 1,
    ):
       
        self.class_name = class_name
        self.start_line = start_line
        self.end_line = end_line
        self.original_chunk_text = original_chunk_text
        self.docstring = docstring
        self.base_classes = base_classes or []
        self.decorators = decorators or []
        self.methods = methods or []
        self.complexity_score = complexity_score
        self.chunk_type = "class"  # Identifier for vector store

    def _get_relative_path(self, file_path: str) -> str:
        """Get project-relative path."""
        try:
            return str(Path(file_path).relative_to(Path.cwd()))
        except ValueError:
            return Path(file_path).name

    def to_payload(self) -> Dict[str, Any]:
        """Convert to VectorDB payload format."""
        return {
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "class_name": self.class_name,
            "function_name": self.class_name,  # For compatibility with queries
            "line_numbers": {"start": self.start_line, "end": self.end_line},
            "original_chunk_text": self.original_chunk_text,
            "chunk_type": self.chunk_type,
            "metadata": {
                "file_extension": self.file_extension,
                "ast_node_type": "class",
                "complexity_score": self.complexity_score,
                "docstring": self.docstring,
                "decorators": self.decorators,
                "base_classes": self.base_classes,
                "methods": self.methods,
                "method_count": len(self.methods),
            },
        }

    def get_enhanced_embedding_text(self) -> str:
        """Generate enhanced text for embeddings using AST metadata.

        Combines class definition with semantic context:
        - Class name
        - Inheritance (base classes)
        - Decorators (e.g., @dataclass)
        - Docstring
        - List of methods (interface)
        - Complexity hint

        This gives the embedding model rich semantic context for class-level queries.
        """
        parts = []

        # Add class identifier
        parts.append(f"Class: {self.class_name}")

        # Add inheritance information
        if self.base_classes:
            parts.append(f"Inherits from: {', '.join(self.base_classes)}")

        # Add decorators
        if self.decorators:
            parts.append(f"Decorators: {', '.join(self.decorators)}")

        # Add docstring
        if self.docstring:
            parts.append(f"Documentation: {self.docstring}")

        # Add methods list (interface)
        if self.methods:
            parts.append(f"Methods: {', '.join(self.methods)}")

        # Add complexity
        if self.complexity_score > 1:
            parts.append(f"Complexity: {self.complexity_score}")

        # Add the actual code (truncated for large classes)
        if len(self.original_chunk_text) > self.chunk_size:
            # For large classes, include definition and summary
            code_preview = (
                self.original_chunk_text[:self.chunk_size]
                + "\\n... (class definition continues) ..."
            )
            parts.append(f"Code:\\n{code_preview}")
        else:
            parts.append(f"Code:\\n{self.original_chunk_text}")

        return "\\n".join(parts)

    def __repr__(self) -> str:
        return f"ClassChunk({self.class_name} @ {self.relative_path}:{self.start_line}-{self.end_line})"


class ASTTextSplitter(TextSplitter):

    def extract_function_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> str:
        """Extract clean function signature from AST node."""
        args = []

        # Regular arguments with type annotations and defaults
        for i, arg in enumerate(node.args.args):
            arg_str = arg.arg

            # Add type annotation if present
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"

            # Add default value if present
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                arg_str += f" = {ast.unparse(node.args.defaults[default_idx])}"

            args.append(arg_str)

        # Handle *args
        if node.args.vararg:
            vararg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(vararg_str)

        # Handle keyword-only arguments
        for i, arg in enumerate(node.args.kwonlyargs):
            kwarg_str = arg.arg
            if arg.annotation:
                kwarg_str += f": {ast.unparse(arg.annotation)}"
            if i < len(node.args.kw_defaults):
                default_val = node.args.kw_defaults[i]
                if default_val is not None:
                    kwarg_str += f" = {ast.unparse(default_val)}"
            args.append(kwarg_str)

        # Handle **kwargs
        if node.args.kwarg:
            kwarg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(kwarg_str)

        # Handle return annotation
        return_annotation = ""
        if node.returns:
            return_annotation = f" -> {ast.unparse(node.returns)}"

        # Build signature
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)}){return_annotation}:"

    def extract_docstring(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Optional[str]:
        """Extract docstring from function node."""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value
        return None

    def extract_decorators(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> List[str]:
        """Extract decorator names from function node."""
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorators.append(ast.unparse(decorator))
            except Exception:
                # Fallback for complex decorators
                decorators.append(str(decorator))
        return decorators

    def calculate_complexity(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Calculate rough cyclomatic complexity."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points increase complexity
            if isinstance(
                child, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)
            ):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def extract_class_docstring(self, node: ast.ClassDef) -> Optional[str]:
        """Extract docstring from class node."""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value
        return None

    def extract_class_decorators(self, node: ast.ClassDef) -> List[str]:
        """Extract decorator names from class node."""
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorators.append(ast.unparse(decorator))
            except Exception:
                decorators.append(str(decorator))
        return decorators

    def extract_base_classes(self, node: ast.ClassDef) -> List[str]:
        """Extract base class names."""
        base_classes = []
        for base in node.bases:
            try:
                base_classes.append(ast.unparse(base))
            except Exception:
                base_classes.append(str(base))
        return base_classes

    def extract_class_methods(self, node: ast.ClassDef) -> List[str]:
        """Extract method names from class."""
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
        return methods

    def calculate_class_complexity(self, node: ast.ClassDef) -> int:
        """Calculate class complexity (sum of method complexities)."""
        complexity = 1
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity += self.calculate_complexity(item)
        return complexity

    """
    Spliting text based on AST
    """

    def split_text(self, text):
        try:
            # Parse the AST
            tree = ast.parse(text)
            source_lines = text.split("\n")

            chunks = []

            # First pass: Extract classes at module level
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    # Extract class text
                    start_line = node.lineno
                    end_line = node.end_lineno if node.end_lineno else start_line

                    # Get the actual class text
                    class_lines = source_lines[start_line - 1 : end_line]
                    original_text = "\n".join(class_lines)

                    # Create class chunk
                    class_chunk = ClassChunk(
                        class_name=node.name,
                        start_line=start_line,
                        end_line=end_line,
                        original_chunk_text=original_text,
                        docstring=self.extract_class_docstring(node),
                        base_classes=self.extract_base_classes(node),
                        decorators=self.extract_class_decorators(node),
                        methods=self.extract_class_methods(node),
                        complexity_score=self.calculate_class_complexity(node),
                    )

                    chunks.append(class_chunk)

            # Second pass: Extract all functions (both module-level and class methods)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Extract function text
                    start_line = node.lineno
                    end_line = node.end_lineno if node.end_lineno else start_line

                    # Get the actual function text
                    function_lines = source_lines[start_line - 1 : end_line]
                    original_text = "\n".join(function_lines)

                    # Create function chunk
                    chunk = FunctionChunk(
                        function_name=node.name,
                        function_signature=self.extract_function_signature(node),
                        start_line=start_line,
                        end_line=end_line,
                        original_chunk_text=original_text,
                        docstring=self.extract_docstring(node),
                        decorators=self.extract_decorators(node),
                        ast_node_type=(
                            "async_function"
                            if isinstance(node, ast.AsyncFunctionDef)
                            else "function"
                        ),
                        complexity_score=self.calculate_complexity(node),
                    )

                    chunks.append(chunk)

            class_count = sum(1 for c in chunks if isinstance(c, ClassChunk))
            func_count = sum(1 for c in chunks if isinstance(c, FunctionChunk))
            return chunks

        except SyntaxError as e:
            logger.error(f"Syntax error in {text}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error chunking {text}: {e}")
            return []
