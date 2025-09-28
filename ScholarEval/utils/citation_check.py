import os
from ..engine.litellm_engine import LLMEngine


def check_and_format_citations(report_file: str, bibliography_file: str, llm_engine: str = "GPT-4.1-nano") -> str:
    """
    Check and format citations in a report using bibliography.
    
    Args:
        report_file: Path to the markdown report file
        bibliography_file: Path to the text bibliography file
        llm_engine: LLM model name to use
    
    Returns:
        Formatted report with proper citations
    """
    
    # Load report and bibliography
    with open(report_file, "r", encoding="utf-8") as f:
        report_content = f.read()
    
    with open(bibliography_file, "r", encoding="utf-8") as f:
        bibliography_content = f.read()
    
    # Initialize LLM engine
    llm = LLMEngine(
        llm_engine_name=llm_engine,
        api_key=os.environ.get("API_KEY_1"),
        api_endpoint=os.environ.get("API_ENDPOINT")
    )
    
    # Create prompt for citation checking and formatting
    prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert academic editor responsible for ensuring proper citation formatting in research proposal review reports. "
                "Your task is to review a report and ensure ALL citation-worthy statements have proper in-text citations "
                "and that a properly formatted References section is included at the end.\n\n"
                
                "CRITICAL REQUIREMENTS:\n"
                "1. PRESERVE CONTENT EXACTLY: Do not change any content, arguments, or wording. Only add citations and formatting. And return the full citatation-checked report.\n"
                "2. CITATION STYLE: Use consistent in-text citations in the format [Author et al., Year](link)\n"
                "3. REFERENCES FORMAT: Author. Title. Year. Venue, with link in format [Title](link)\n"
                "4. MARKDOWN LINKS: All citations and references must include working markdown links\n"
                "5. COMPLETENESS: Every factual claim, finding, or statement that references prior work must have a citation\n\n"
                
                "CITATION PLACEMENT RULES:\n"
                "- Add in-text citations immediately after statements that reference prior work\n"
                "- Multiple citations should be separated by semicolons: [Author1 et al., 2020](link1); [Author2 et al., 2021](link2)\n"
                "- Place citations before punctuation marks\n\n"
                
                "REFERENCES SECTION:\n"
                "- Add a '## References' section at the end if not present\n"
                "- Format each reference as: Author. [Title](link). Year. Venue.\n"
                "- Alphabetize references by first author's last name\n"
                "- Only include references that are actually cited in the text. Drop all other references that are not present as an in-text citation.\n\n"
                "- In each reference, if any of the components (Author, Title, Year, Venue) are missing (e.g. No venue, etc) just skip them, do not out these placeholders in the reference.\n\n"
                "- Make sure that each reference includes a working markdown link to the source.\n\n"
                
                "OUTPUT REQUIREMENTS:\n"
                "- Return the complete report with proper citations added\n"
                "- Maintain exact original content and structure\n"
                "- Ensure all links are in proper markdown format\n"
                "- Do not add explanatory text or meta-commentary\n\n"
                
                "Remember: Your role is purely editorial - add citations and formatting without changing the content."
            )
        },
        {
            "role": "user", 
            "content": (
                f"Research Report:\n\n{report_content}\n\n"
                f"Available Bibliography:\n\n{bibliography_content}\n\n"
                "Please review this report and add proper in-text citations for all citation-worthy statements, "
                "ensuring that a properly formatted References section is included at the end. "
                "Remember to preserve the exact content while only adding proper citations and formatting. "
                "All citations must use markdown links in the specified format."
            )
        }
    ]
    
    # Get formatted response from LLM
    response, _, _ = llm.respond(prompt, temperature=0.1)
    return response.strip()


def check_citations_from_strings(report_content: str, bibliography_content: str, llm_engine: str = "GPT-4.1-nano") -> str:
    """
    Check and format citations given report and bibliography as strings.
    
    Args:
        report_content: The report content as string
        bibliography_content: The bibliography content as string
        llm_engine: LLM model name to use
    
    Returns:
        Formatted report with proper citations
    """
    
    # Initialize LLM engine
    llm = LLMEngine(
        llm_engine_name=llm_engine,
        api_key=os.environ.get("API_KEY_1"),
        api_endpoint=os.environ.get("API_ENDPOINT")
    )
    
    # Create prompt for citation checking and formatting
    prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert academic editor responsible for ensuring proper citation formatting in research proposal review reports. "
                "Your task is to review a report and ensure ALL citation-worthy statements have proper in-text citations "
                "and that a properly formatted References section is included at the end.\n\n"
                
                "CRITICAL REQUIREMENTS:\n"
                "1. PRESERVE CONTENT EXACTLY: Do not change any content, arguments, or wording. Only add citations and formatting. And return the full citatation-checked report.\n"
                "2. CITATION STYLE: Use consistent in-text citations in the format [Author et al., Year](link)\n"
                "3. REFERENCES FORMAT: Author. Title. Year. Venue, with link in format [Title](link)\n"
                "4. MARKDOWN LINKS: All citations and references must include working markdown links\n"
                "5. COMPLETENESS: Every factual claim, finding, or statement that references prior work must have a citation\n\n"
                
                "CITATION PLACEMENT RULES:\n"
                "- Add in-text citations immediately after statements that reference prior work\n"
                "- Multiple citations should be separated by semicolons: [Author1 et al., 2020](link1); [Author2 et al., 2021](link2)\n"
                "- Place citations before punctuation marks\n\n"
                
                "REFERENCES SECTION:\n"
                "- Add a '## References' section at the end if not present\n"
                "- Format each reference as: Author. [Title](link). Year. Venue.\n"
                "- Alphabetize references by first author's last name\n"
                "- Only include references that are actually cited in the text. Drop all other references that are not present as an in-text citation.\n\n"
                "- In each reference, if any of the components (Author, Title, Year, Venue) are missing (e.g. No venue, etc) just skip them, do not out these placeholders in the reference.\n\n"
                "- Make sure that each reference includes a working markdown link to the source.\n\n"
                "- If there are duplicate entries in the reference section, keep only one. "
                
                "OUTPUT REQUIREMENTS:\n"
                "- Return the complete report with proper citations added\n"
                "- Maintain exact original content and structure\n"
                "- Ensure all links are in proper markdown format\n"
                "- Do not add explanatory text or meta-commentary\n\n"
                
                "Remember: Your role is purely editorial - add citations and formatting without changing the content."
            )
        },
        {
            "role": "user", 
            "content": (
                f"Research Report:\n\n{report_content}\n\n"
                f"Available Bibliography:\n\n{bibliography_content}\n\n"
                "Please review this report and add proper in-text citations for all citation-worthy statements, "
                "ensuring that a properly formatted References section is included at the end. "
                "Remember to preserve the exact content while only adding proper citations and formatting. "
                "All citations must use markdown links in the specified format."
            )
        }
    ]
    
    # Get formatted response from LLM
    response, _, _ = llm.respond(prompt, temperature=0.1)
    return response.strip()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check and format citations in a research report.")
    parser.add_argument("--report", required=True, help="Path to the markdown report file.")
    parser.add_argument("--bibliography", required=True, help="Path to the text bibliography file.")
    parser.add_argument("--llm_engine", default="GPT-4.1-nano", help="LLM model name to use.")
    parser.add_argument("--output", required=True, help="Path to save the formatted report.")
    
    args = parser.parse_args()
    
    # Process the citation checking
    formatted_report = check_and_format_citations(args.report, args.bibliography, args.llm_engine)
    
    # Save the formatted report
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(formatted_report)
    
    print(f"Formatted report saved to {args.output}")