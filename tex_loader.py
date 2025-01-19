import os
import re
from typing import Iterator, List, Dict
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader

class TeXDirectoryLoader(BaseLoader):
    """A document loader for TeX files that can handle both single files and directories."""

    def __init__(
        self, 
        path: str, 
        debug: bool = False, 
        recursive: bool = True,
        file_pattern: str = "*.tex"
    ) -> None:
        """Initialize the loader with a file or directory path.

        Args:
            path: Path to a .tex file or directory containing .tex files
            debug: Whether to print debug messages
            recursive: Whether to recursively search subdirectories
            file_pattern: Pattern to match tex files (default: "*.tex")
        """
        self.path = path
        self.debug = debug
        self.recursive = recursive
        self.file_pattern = file_pattern

    def _log(self, message: str) -> None:
        """Print debug messages if debug mode is on."""
        if self.debug:
            print(f"[DEBUG] {message}")

    def _find_tex_files(self) -> List[str]:
        """Find all TeX files in the given path."""
        tex_files = []
        
        if os.path.isfile(self.path):
            if self.path.endswith('.tex'):
                tex_files.append(self.path)
        else:
            # Walk through directory
            for root, _, files in os.walk(self.path):
                if not self.recursive and root != self.path:
                    continue
                    
                for file in files:
                    if file.endswith('.tex'):
                        full_path = os.path.join(root, file)
                        tex_files.append(full_path)
        
        self._log(f"Found {len(tex_files)} TeX files")
        return tex_files

    def _extract_sections(self, content: str, file_path: str) -> List[Dict]:
        """Extract sections and their content from TeX file."""
        sections = []
        
        # First, let's handle the preamble
        doc_start = content.find("\\begin{document}")
        if doc_start == -1:
            self._log(f"No \\begin{{document}} found in {file_path}")
            return [{"title": "Main", "content": content, "level": 0}]
        
        # Extract preamble
        preamble = content[:doc_start].strip()
        if preamble:
            sections.append({
                "title": "Preamble",
                "content": preamble,
                "level": 0
            })
            self._log(f"Added preamble section of length {len(preamble)} in {file_path}")
        
        # Get main content
        doc_end = content.find("\\end{document}")
        if doc_end == -1:
            main_content = content[doc_start + len("\\begin{document}"):]
        else:
            main_content = content[doc_start + len("\\begin{document}"):doc_end]
        
        # Find all sections
        section_pattern = r"\\((?:sub)*section\*?)\{([^}]+)\}"
        current_pos = 0
        current_title = None
        current_level = 0
        
        for match in re.finditer(section_pattern, main_content):
            # Get the content from current_pos to this section start
            if current_pos > 0 and current_title:  # If not the first section
                section_content = main_content[current_pos:match.start()].strip()
                if section_content:
                    sections.append({
                        "title": current_title,
                        "content": section_content,
                        "level": current_level
                    })
                    self._log(f"Added section '{current_title}' with content length {len(section_content)} in {file_path}")
            
            # Update current section info
            current_level = match.group(1).count("sub")
            current_title = match.group(2)
            current_pos = match.end()
        
        # Add the last section
        if current_pos < len(main_content) and current_title:
            last_content = main_content[current_pos:].strip()
            if last_content:
                sections.append({
                    "title": current_title,
                    "content": last_content,
                    "level": current_level
                })
                self._log(f"Added final section '{current_title}' with content length {len(last_content)} in {file_path}")
        
        # If no sections found, treat entire content as one section
        if not sections:
            self._log(f"No sections found in {file_path}, treating as single document")
            sections.append({
                "title": "Main Content",
                "content": main_content.strip(),
                "level": 0
            })
        
        return sections

    def _clean_tex_commands(self, content: str) -> str:
        """Remove or simplify TeX commands from content."""
        if not content:
            return content
            
        # Keep track of content length before and after cleaning
        original_length = len(content)
        
        # Basic command cleaning
        cleaners = [
            (r"\\begin\{[^}]+\}", ""),  # Remove begin commands
            (r"\\end\{[^}]+\}", ""),    # Remove end commands
            (r"\\item\s*", "- "),       # Convert items to bullet points
            (r"\\textbf\{([^}]+)\}", r"\1"),  # Remove bold markers but keep content
            (r"\\textit\{([^}]+)\}", r"\1"),  # Remove italic markers but keep content
            (r"\\href\{[^}]+\}\{([^}]+)\}", r"\1"),  # Keep link text only
            (r"\\[a-zA-Z]+\s", " "),    # Remove simple commands
        ]
        
        for pattern, replacement in cleaners:
            content = re.sub(pattern, replacement, content)
        
        # Clean up extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        cleaned_length = len(content)
        self._log(f"Cleaned content from {original_length} to {cleaned_length} characters")
        
        return content

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load TeX files and yield documents by section."""
        tex_files = self._find_tex_files()
        
        for file_path in tex_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    self._log(f"Read file {file_path} of length {len(content)}")
            except Exception as e:
                self._log(f"Error reading file {file_path}: {e}")
                continue
            
            sections = self._extract_sections(content, file_path)
            self._log(f"Extracted {len(sections)} sections from {file_path}")
            
            for i, section in enumerate(sections):
                cleaned_content = self._clean_tex_commands(section["content"])
                
                if not cleaned_content.strip():
                    self._log(f"Warning: Empty content for section {section['title']} in {file_path}")
                    continue
                    
                metadata = {
                    "source": file_path,
                    "section": section["title"],
                    "level": section["level"],
                    "section_number": i,
                    "file_name": os.path.basename(file_path),
                    "original_length": len(section["content"]),
                    "cleaned_length": len(cleaned_content)
                }
                
                yield Document(
                    page_content=cleaned_content,
                    metadata=metadata
                )

    def load(self) -> List[Document]:
        """Load all documents at once."""
        return list(self.lazy_load())