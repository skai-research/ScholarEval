import requests
import json
from typing import List, Dict, Any, Optional
import time
import re
import ast
import xml.etree.ElementTree as ET

class StringUtils(): 
    """
    A class for string utilities 
    """

    # def __init__(self, api_key: str):

    def extract_json_output(self, response, extra=None, n_docs_used=0):
        import re, json

        def clean_json_string(text):
            # Escape unescaped backslashes
            text = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', text)
            return text

        try:
            if response.startswith("```json"):
                match = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", response, re.DOTALL)
                json_str = match.group(1)
            else:
                json_str = response.strip()
            json_str = clean_json_string(json_str)
            extracted_resp = json.loads(json_str)

            # If the extracted object is a list, log and extract first item
            if isinstance(extracted_resp, list):
                print("⚠️ extract_json_output: Extracted JSON is a list, using first element.")
                extracted_resp = extracted_resp[0]  # or handle this more carefully if needed

        except Exception as e:
            print(f"❌ Skipping response due to JSON parse error:")
            print(response)
            print("Error:", e)
            extracted_resp = None

        return extracted_resp

    def extract_jsonl_output(self, response, extra=None, n_docs_used=0):
        import re, json

        def clean_json_string(text):
            # Escape unescaped backslashes
            text = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', text)
            return text

        try:
            # Extract JSONL from code block
            if "```jsonl" in response:
                match = re.search(r"```jsonl\s*(.*?)\s*```", response, re.DOTALL)
                if match:
                    jsonl_str = match.group(1).strip()
                else:
                    jsonl_str = response.strip()
            else:
                jsonl_str = response.strip()

            jsonl_objects = []
            
            # First, try to parse as proper JSONL (one JSON object per line)
            lines = jsonl_str.split('\n')
            
            # Check if we have multi-line JSON objects (improper JSONL format)
            if any(line.strip() in ['{', '}'] for line in lines):
                # Try to reconstruct JSON objects from multi-line format
                current_obj = ""
                brace_count = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    current_obj += line
                    
                    # Count braces to detect complete JSON objects
                    brace_count += line.count('{') - line.count('}')
                    
                    if brace_count == 0 and current_obj:
                        try:
                            current_obj = clean_json_string(current_obj)
                            json_obj = json.loads(current_obj)
                            jsonl_objects.append(json_obj)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Skipping invalid JSON object: {current_obj}")
                            print(f"Error: {e}")
                            continue
                        current_obj = ""
            else:
                # Parse as proper JSONL (one JSON object per line)
                for line in lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            line = clean_json_string(line)
                            json_obj = json.loads(line)
                            jsonl_objects.append(json_obj)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Skipping invalid JSON line: {line}")
                            print(f"Error: {e}")
                            continue

            return jsonl_objects

        except Exception as e:
            print(f"❌ Skipping response due to JSONL parse error:")
            print(response)
            print("Error:", e)
            return []


    def extract_python_list(self, response):
        try:
            match = re.search(r"\[\s*(.*?)\s*\]", response, re.DOTALL)
            list_content = "[" + match.group(1).strip() + "]"
            clean_resp = ast.literal_eval(list_content)
        except Exception as e:
            print(f"Error parsing methods: {e}")
            print('response:', response)
            clean_resp = []
        return clean_resp
    

class GrobidXMLParser:
    def __init__(self, xml_content: str):
        """
        Initialize the parser with XML content.
        
        Args:
            xml_content: The XML content as a string
        """
        self.xml_content = xml_content
        self.root = None
        self.namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
        self._parse_xml()
    
    def _parse_xml(self):
        """Parse the XML content and set up the root element."""
        try:
            self.root = ET.fromstring(self.xml_content)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            # Try wrapping in a root element if it's a fragment
            try:
                wrapped_xml = f'<root xmlns="http://www.tei-c.org/ns/1.0">{self.xml_content}</root>'
                self.root = ET.fromstring(wrapped_xml)
            except ET.ParseError:
                raise ValueError("Unable to parse XML content")
    
    def extract_sections(self) -> List[Dict]:
        """
        Extract all sections with their headers and content.
        
        Returns:
            List of dictionaries containing section information
        """
        sections = []
        
        # Find all div elements (sections)
        divs = self.root.findall('.//tei:div', self.namespace)
        
        for div in divs:
            section = {}
            
            # Extract section number and title from head
            head = div.find('tei:head', self.namespace)
            if head is not None:
                section['header'] = head.text or ""
                section['section_number'] = head.get('n', '')
            else:
                section['header'] = ""
                section['section_number'] = ""
            
            # Extract all paragraphs in this section
            paragraphs = []
            for p in div.findall('tei:p', self.namespace):
                paragraph_text = self._extract_text_with_refs(p)
                if paragraph_text.strip():
                    paragraphs.append(paragraph_text.strip())
            
            section['paragraphs'] = paragraphs
            section['full_text'] = ' '.join(paragraphs)
            
            sections.append(section)
        
        return sections
    
    def _extract_text_with_refs(self, element) -> str:
        """
        Extract text from an element, handling references and maintaining readability.
        
        Args:
            element: XML element to extract text from
            
        Returns:
            Cleaned text string
        """
        text_parts = []
        
        # Get the element's own text
        if element.text:
            text_parts.append(element.text)
        
        # Process child elements
        for child in element:
            if child.tag.endswith('}ref'):
                # Handle references - extract the citation
                ref_text = child.text or ""
                ref_target = child.get('target', '')
                ref_type = child.get('type', '')
                
                # Format reference based on type
                if ref_type == 'bibr':
                    text_parts.append(f"({ref_text})")
                else:
                    text_parts.append(ref_text)
            else:
                # For other elements, just get the text
                text_parts.append(child.text or "")
            
            # Add tail text (text after the child element)
            if child.tail:
                text_parts.append(child.tail)
        
        return ''.join(text_parts)
    
    def extract_references(self) -> List[Dict]:
        """
        Extract all bibliographic references from the document.
        
        Returns:
            List of reference dictionaries
        """
        references = []
        
        # Find all ref elements with type="bibr"
        refs = self.root.findall('.//tei:ref[@type="bibr"]', self.namespace)
        
        for ref in refs:
            ref_data = {
                'text': ref.text or "",
                'target': ref.get('target', ''),
                'type': ref.get('type', '')
            }
            references.append(ref_data)
        
        return references
    
    def extract_figures(self) -> List[Dict]:
        """
        Extract figure references and captions.
        
        Returns:
            List of figure dictionaries
        """
        figures = []
        
        # Find all ref elements with type="figure"
        fig_refs = self.root.findall('.//tei:ref[@type="figure"]', self.namespace)
        
        for fig_ref in fig_refs:
            figure_data = {
                'reference_text': fig_ref.text or "",
                'target': fig_ref.get('target', ''),
                'type': fig_ref.get('type', '')
            }
            figures.append(figure_data)
        
        return figures
    
    def get_full_text(self) -> str:
        """
        Extract all text content from the document.
        
        Returns:
            Full document text as a string
        """
        sections = self.extract_sections()
        full_text_parts = []
        
        for section in sections:
            if section['header']:
                full_text_parts.append(f"\n{section['header']}\n")
            full_text_parts.append(section['full_text'])
        
        return '\n'.join(full_text_parts)
    
    def to_dict(self) -> Dict:
        """
        Convert the entire document to a structured dictionary.
        
        Returns:
            Dictionary containing all extracted information
        """
        return {
            'sections': self.extract_sections(),
            'references': self.extract_references(),
            'figures': self.extract_figures(),
            'full_text': self.get_full_text()
        }

    def extract_bibliography(self) -> List[Dict]:
        """
        Extract full bibliography entries from the document.
        
        Returns:
            List of bibliography entry dictionaries
        """
        bibliography = []
        
        # Find all bibl elements in the bibliography section
        bibl_entries = self.root.findall('.//tei:listBibl/tei:biblStruct', self.namespace)
        
        for bibl in bibl_entries:
            entry = {
                'id': bibl.get('{http://www.w3.org/XML/1998/namespace}id', ''),
                'title': '',
                'authors': [],
                'year': '',
                'venue': '',
                'pages': '',
                'doi': ''
            }
            
            # Extract title
            title_elem = bibl.find('.//tei:title[@level="a"]', self.namespace)
            if title_elem is not None:
                entry['title'] = title_elem.text or ""
            
            # Extract authors
            authors = bibl.findall('.//tei:author/tei:persName', self.namespace)
            for author in authors:
                forename = author.find('tei:forename', self.namespace)
                surname = author.find('tei:surname', self.namespace)
                
                author_name = ""
                if forename is not None and forename.text:
                    author_name += forename.text + " "
                if surname is not None and surname.text:
                    author_name += surname.text
                
                if author_name.strip():
                    entry['authors'].append(author_name.strip())
            
            # Extract year
            date_elem = bibl.find('.//tei:date', self.namespace)
            if date_elem is not None:
                entry['year'] = date_elem.get('when', '') or date_elem.text or ""
            
            # Extract venue
            venue_elem = bibl.find('.//tei:title[@level="j"]', self.namespace) or \
                        bibl.find('.//tei:title[@level="m"]', self.namespace)
            if venue_elem is not None:
                entry['venue'] = venue_elem.text or ""
            
            # Extract pages
            pages_elem = bibl.find('.//tei:biblScope[@unit="page"]', self.namespace)
            if pages_elem is not None:
                entry['pages'] = pages_elem.text or ""
            
            # Extract DOI
            doi_elem = bibl.find('.//tei:idno[@type="DOI"]', self.namespace)
            if doi_elem is not None:
                entry['doi'] = doi_elem.text or ""
            
            bibliography.append(entry)
        
        return bibliography
    
    def find_related_work_section(self) -> Optional[Dict]:
        """
        Find the related work section using common section titles.
        
        Returns:
            Dictionary containing the related work section or None if not found
        """
        sections = self.extract_sections()
        
        # Common patterns for related work sections
        related_work_patterns = [
            r'related\s+work',
            r'background',
            r'literature\s+review',
            r'prior\s+work',
            r'previous\s+work',
            r'state\s+of\s+the\s+art',
            r'existing\s+approaches',
            r'survey'
        ]
        
        for section in sections:
            header_lower = section['header'].lower()
            for pattern in related_work_patterns:
                if re.search(pattern, header_lower):
                    return section
        
        return None
    
    def extract_references_from_section(self, section_text: str) -> List[str]:
        """
        Extract reference IDs mentioned in a section.
        
        Args:
            section_text: Text content of the section (not used - we parse XML directly)
            
        Returns:
            List of reference IDs (e.g., ['b0', 'b1', 'b2'])
        """
        # Since we have the full XML structure, we should extract reference IDs 
        # directly from the related work section XML rather than parsing text
        related_work_section = self.find_related_work_section()
        if not related_work_section:
            return []
        
        # Find the related work div in the XML
        divs = self.root.findall('.//tei:div', self.namespace)
        related_div = None
        
        for div in divs:
            head = div.find('tei:head', self.namespace)
            if head is not None and head.text and 'related' in head.text.lower():
                related_div = div
                break
        
        if not related_div:
            return []
        
        # Extract all reference targets from this div
        ref_ids = []
        refs = related_div.findall('.//tei:ref[@type="bibr"]', self.namespace)
        
        for ref in refs:
            target = ref.get('target', '')
            if target.startswith('#'):
                ref_id = target[1:]  # Remove the '#' prefix
                if ref_id and ref_id not in ref_ids:
                    ref_ids.append(ref_id)
        
        return ref_ids

    def extract_elife_assessment_and_reviews(self) -> Dict[str, Any]:
        """
        Extract eLife Assessment and reviewer sections from the document.
        
        Returns:
            Dictionary containing assessment and reviews
        """
        result = {
            'elife_assessment': None,
            'reviews': []
        }
        
        sections = self.extract_sections()
        
        for section in sections:
            header_lower = section['header'].lower()
            
            # Check for eLife Assessment
            if 'elife assessment' in header_lower:
                result['elife_assessment'] = {
                    'header': section['header'],
                    'content': section['full_text'],
                    'paragraphs': section['paragraphs']
                }
            
            # Check for reviewer sections
            elif 'reviewer' in header_lower:
                result['reviews'].append({
                    'header': section['header'],
                    'content': section['full_text'],
                    'paragraphs': section['paragraphs']
                })
        
        return result

    def remove_assessment_and_reviews_from_xml(self) -> str:
        """
        Remove eLife Assessment and reviewer sections from the XML.
        
        Returns:
            Modified XML content as string
        """
        # Create a copy of the root to modify
        import copy
        modified_root = copy.deepcopy(self.root)
        
        # Find and remove divs containing eLife Assessment or reviewer sections
        divs_to_remove = []
        divs = modified_root.findall('.//tei:div', self.namespace)
        
        for div in divs:
            head = div.find('tei:head', self.namespace)
            if head is not None and head.text:
                header_lower = head.text.lower()
                if 'elife assessment' in header_lower or 'reviewer' in header_lower:
                    divs_to_remove.append(div)
        
        # Remove the identified divs
        for div in divs_to_remove:
            parent = div.find('..')
            if parent is not None:
                parent.remove(div)
            else:
                # If no parent found, try to find it in the tree
                for elem in modified_root.iter():
                    if div in elem:
                        elem.remove(div)
                        break
        
        # Convert back to string
        return ET.tostring(modified_root, encoding='unicode')

    def remove_figures_and_tables_from_xml(self) -> str:
        """
        Remove figures and tables from the XML.
        
        Returns:
            Modified XML content as string
        """
        # Create a copy of the root to modify
        import copy
        modified_root = copy.deepcopy(self.root)
        
        # Remove figure elements
        figures = modified_root.findall('.//tei:figure', self.namespace)
        for fig in figures:
            parent = fig.find('..')
            if parent is not None:
                parent.remove(fig)
            else:
                # Find parent in tree
                for elem in modified_root.iter():
                    if fig in elem:
                        elem.remove(fig)
                        break
        
        # Remove table elements
        tables = modified_root.findall('.//tei:table', self.namespace)
        for table in tables:
            parent = table.find('..')
            if parent is not None:
                parent.remove(table)
            else:
                # Find parent in tree
                for elem in modified_root.iter():
                    if table in elem:
                        elem.remove(table)
                        break
        
        # Remove figure and table references
        refs_to_remove = []
        refs = modified_root.findall('.//tei:ref', self.namespace)
        for ref in refs:
            ref_type = ref.get('type', '')
            if ref_type in ['figure', 'table']:
                refs_to_remove.append(ref)
        
        for ref in refs_to_remove:
            parent = ref.find('..')
            if parent is not None:
                parent.remove(ref)
            else:
                # Find parent in tree
                for elem in modified_root.iter():
                    if ref in elem:
                        elem.remove(ref)
                        break
        
        # Convert back to string
        return ET.tostring(modified_root, encoding='unicode')

    def extract_clean_sections_for_text(self) -> List[Dict]:
        """
        Extract sections excluding eLife Assessment, reviews, figures, and tables.
        
        Returns:
            List of clean section dictionaries
        """
        sections = self.extract_sections()
        clean_sections = []
        
        for section in sections:
            header_lower = section['header'].lower()
            
            # Skip eLife Assessment and reviewer sections
            if 'elife assessment' in header_lower or 'reviewer' in header_lower:
                continue
                
            clean_sections.append(section)
        
        return clean_sections

    def generate_clean_text(self) -> str:
        """
        Generate clean text from the document excluding assessments, reviews, figures, and tables.
        
        Returns:
            Clean text content as string
        """
        # First, get the XML without assessments and reviews
        xml_without_assessment = self.remove_assessment_and_reviews_from_xml()
        
        # Create a new parser with the modified XML
        temp_parser = GrobidXMLParser(xml_without_assessment)
        
        # Get the modified XML without figures and tables
        xml_clean = temp_parser.remove_figures_and_tables_from_xml()
        
        # Create final parser and extract text
        final_parser = GrobidXMLParser(xml_clean)
        sections = final_parser.extract_sections()
        
        text_parts = []
        for section in sections:
            if section['header'].strip():
                text_parts.append(f"\n# {section['header']}\n")
            if section['full_text'].strip():
                text_parts.append(section['full_text'])
                text_parts.append('\n')
        
        return '\n'.join(text_parts).strip()


# Example usage:
def parse_grobid_file(file_path: str) -> Dict:
    """
    Parse a GROBID XML file and return structured data.
    
    Args:
        file_path: Path to the XML file
        
    Returns:
        Dictionary containing parsed document data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    parser = GrobidXMLParser(xml_content)
    return parser.to_dict()