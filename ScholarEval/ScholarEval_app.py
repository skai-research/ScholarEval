import streamlit as st
import os
import tempfile
import subprocess
import json
from pathlib import Path
import time
import PyPDF2

def ensure_demo_data_dir():
    """Create and return the demo_data directory path."""
    demo_data_path = Path("demo_data")
    demo_data_path.mkdir(exist_ok=True)
    return demo_data_path

def monitor_progress_with_details(process, progress_file, status_placeholder, progress_bar, base_progress, progress_range):
    """Monitor a subprocess and update progress based on a progress file with enhanced details."""
    last_progress = base_progress
    progress_info_container = st.empty()  # For progress info
    
    while process.poll() is None:
        time.sleep(1)
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                current = progress_data.get('current', 0)
                total = progress_data.get('total', 1)
                details = progress_data.get('details', '')
                status = progress_data.get('status', '')
                
                if total > 0:
                    step_progress = (current / total) * progress_range
                    new_progress = base_progress + step_progress
                    progress_bar.progress(min(int(new_progress), 100))
                    last_progress = new_progress
                
                if status == 'processing' and 'method' in details.lower():
                    status_placeholder.text(f"{details}")
                elif status == 'processing' and 'snippet' in details.lower():
                    status_placeholder.text(f"🔍 {details}")
                    if 'found' in details.lower() and 'snippet' in details.lower():
                        progress_info_container.info(f"Literature search progress: {details}")
                elif 'downloading' in details.lower() or status == 'Downloading all pdfs':
                    status_placeholder.text(f"Downloading papers to read...")
                    if current > 0:
                        progress_info_container.info(f"📄 Downloaded {current} PDFs for processing")
                elif 'extracting' in details.lower() or 'grobid' in details.lower():
                    status_placeholder.text(f"Reading papers ...")
                    if current > 0:
                        ok = 1
                elif details:
                    status_placeholder.text(f"{details}")
                    
            except (json.JSONDecodeError, FileNotFoundError):
                pass
    
    return last_progress

def display_structured_review(review_text, review_type="Review"):
    """Parse and display structured review with timing."""
    try:
        review_data = json.loads(review_text)
        if all(key in review_data for key in ['strengths', 'weaknesses', 'actionable_summary', 'score']):
            st.markdown(f"### {review_type} Results")
            
            st.markdown("#### Strengths")
            st.markdown(review_data['strengths'])
            
            st.markdown("#### Weaknesses")
            st.markdown(review_data['weaknesses'])
            
            st.markdown("#### Actionable Summary")
            st.markdown(review_data['actionable_summary'])
            
            st.markdown("#### Overall Score")
            st.markdown(f"**{review_data['score']}**")
            
            return True
    except (json.JSONDecodeError, KeyError):
        pass
    
    st.markdown(f"### {review_type} Results")
    st.markdown(review_text)
    return False

def display_raw_json(data, title):
    """Display raw JSON data for debugging purposes."""
    if data:
        with st.expander(f"{title}", expanded=False):
            st.json(data)
    else:
        st.warning(f"No {title.lower()} found")

st.set_page_config(
    page_title="ScholarEval: Research Idea Evaluation Grounded in Literature",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 ScholarEval: Research Idea Evaluator Grounded in Literature")

st.sidebar.header("Configuration")
llm_model = st.sidebar.selectbox(
    "Select LLM Model",
    ["GPT-4.1", "GPT-4o", "GPT-4.1-mini", "GPT-4.1-nano", "Anthropic Claude 4 Sonnet", "Anthropic Claude 3.5 Haiku"],
    index=0
)

api_keys_status = st.sidebar.expander("API Keys Status")
required_keys = ["API_KEY", "S2_API_KEY"]
for key in required_keys:
    status = "✅" if os.environ.get(key) else "❌"
    api_keys_status.write(f"{key}: {status}")

# Cost computation option - currently disabled
enable_cost_computation = st.sidebar.checkbox(
    "Enable cost computation", 
    value=False,
    help="Track LLM usage costs during evaluation"
)
litellm_name = None

st.sidebar.markdown("---")
st.sidebar.markdown("### Runs Storage")
demo_data_path = Path("demo_data")
if demo_data_path.exists():
    demo_folders = [f for f in demo_data_path.iterdir() if f.is_dir()]
    if demo_folders:
        st.sidebar.write(f"**Location:** `{demo_data_path.absolute()}`")
        st.sidebar.write(f"**Existing runs:** {len(demo_folders)}")
        with st.sidebar.expander("View Runs", expanded=False):
            for folder in sorted(demo_folders, reverse=True)[:5]:
                st.write(f"• {folder.name}")
            if len(demo_folders) > 5:
                st.write(f"... and {len(demo_folders) - 5} more")
    else:
        st.sidebar.write("**Location:** Will be created at `demo_data/`")
        st.sidebar.write("*No demo runs yet*")
else:
    st.sidebar.write("**Location:** Will be created at `demo_data/`")
    st.sidebar.write("*No demo runs yet*")

st.sidebar.markdown("*All intermediate files will be saved here for inspection*")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Research Idea")
    
    # Option 1: Text input
    research_plan_text = st.text_area(
        "Paste your research idea here:",
        height=300,
        placeholder="Enter your research idea including problem statement, methodology, and experimental design..."
    )
    
    st.markdown("**OR**")
    
    # Option 2: File upload
    uploaded_file = st.file_uploader(
        "Upload research plan file",
        type=['txt', 'md', 'pdf'],
        help="Upload a text file containing your research plan"
    )
    
    # Cutoff date input
    st.markdown("**Optional: Literature Search Cutoff Date**")
    cutoff_date = st.date_input(
        "Cutoff date (YYYY-MM-DD)",
        value=None,
        help="Only search for papers published on or before this date. Leave empty to search all papers.",
        format="YYYY-MM-DD"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            research_plan_text = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "text/markdown":
            research_plan_text = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() or ""
                research_plan_text = pdf_text
            except Exception as e:
                st.error(f"Could not process PDF file: {e}")
                research_plan_text = ""
        else:
            st.warning("Unsupported file type. Please use .txt, .md, or .pdf.")
with col2:
    st.header("Evaluation Results")
    tabs = st.tabs(["Soundness Review", "Contribution Review"])
    can_generate = bool(research_plan_text.strip())
    with tabs[0]:
        st.subheader("Soundness Review")
        if st.button("Generate Soundness Review", disabled=not can_generate, key="soundness_btn"):
            st.info("Running soundness review pipeline...")
            
            try:
                start_time = time.time()
                
                # Create demo_data directory for persistent storage
                demo_data_path = ensure_demo_data_dir()
                soundness_dir = demo_data_path / f"soundness_{int(time.time())}"
                soundness_dir.mkdir(exist_ok=True)
                
                input_file = soundness_dir / "research_plan.txt"
                with open(input_file, 'w') as f:
                    f.write(research_plan_text)
                
                st.info(f"Saving intermediate results to: `{soundness_dir}`")
                
                # Set up environment - always run from root directory where packages are installed
                env = os.environ.copy()
                # Add current directory to Python path for local imports
                import sys
                working_dir = "."  # Always use root directory
                current_python_path = ":".join(sys.path)
                env["PYTHONPATH"] = f"{os.path.abspath('.')}:{current_python_path}:{env.get('PYTHONPATH', '')}"

                # Pipeline steps
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                # Step 1: Extract methods
                status_placeholder.text("Extracting methods from research plan...")
                progress_bar.progress(15)
                st.info("Identifying research methods...")
                cmd1 = ["python", "-m", "ScholarEval.soundness.extract_methods", "--input_file", str(input_file), "--output_file", str(soundness_dir / "methods.json"), "--llm_engine_name", "GPT-4.1-nano", "--cost_log_file", str(soundness_dir / "soundness_costs.jsonl")]
                if litellm_name:
                    cmd1.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd1, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Method extraction failed: {result.stderr}")
                    st.stop()
                st.success("Methods extracted successfully!")
                
                # Display extracted methods
                try:
                    with open(soundness_dir / "methods.json", 'r') as f:
                            methods_data = json.load(f)
                            display_raw_json(methods_data, "Extracted Methods")
                except Exception as e:
                    st.warning(f"Could not display methods: {e}")

                # Step 3: Make queries
                status_placeholder.text("Generating search queries...")
                progress_bar.progress(40)
                st.info("Creating targeted queries for literature search...")
                cmd3 = ["python", "-m", "ScholarEval.soundness.make_queries", "--research_plan", str(input_file), "--methods_file", str(soundness_dir / "methods.json"), "--output_file", str(soundness_dir / "queries.json"), "--llm_engine_name", "GPT-4.1-nano", "--cost_log_file", str(soundness_dir / "soundness_costs.jsonl")]
                if litellm_name:
                    cmd3.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd3, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Query generation failed: {result.stderr}")
                    st.stop()
                st.success("Search queries generated successfully!")
                
                # Display generated queries
                try:
                    with open(soundness_dir / "queries.json", 'r') as f:
                        queries_data = json.load(f)
                    display_raw_json(queries_data, "Generated Search Queries")
                except Exception as e:
                    st.warning(f"Could not display queries: {e}")

                # Step 4: Snippet search with progress monitoring
                status_placeholder.text("Searching for relevant literature...")
                progress_bar.progress(50)
                st.info("Finding relevant academic snippets and papers...")
                
                # Set up progress monitoring for snippet search
                progress_file = soundness_dir / "snippet_progress.json"
                cmd4 = ["python", "-m", "ScholarEval.soundness.snippet_search", "--queries_file", str(soundness_dir / "queries.json"), "--methods_file", str(soundness_dir / "methods.json"), "--output_file", str(soundness_dir / "snippet"), "--pdf_dir", str(soundness_dir / "pdfs"), "--progress_file", str(progress_file)]
                if cutoff_date:
                    cmd4.extend(["--cutoff_date", str(cutoff_date)])
                
                # Start the subprocess for snippet search
                process = subprocess.Popen(cmd4, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=working_dir, env=env)
                
                # Monitor progress with detailed updates
                final_progress = monitor_progress_with_details(process, progress_file, status_placeholder, progress_bar, 50, 20)
                
                # Wait for completion and check result
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    st.error(f"Snippet search failed: {stderr}")
                    st.stop()
                st.success("Literature search completed successfully!")
                
                progress_bar.progress(70)

                # Step 5: Compare methods
                status_placeholder.text("Analyzing existing methods in the literature...")
                progress_bar.progress(80)
                st.info("Analyzing soundness against existing literature...")
                cmd5 = ["python", "-m", "ScholarEval.soundness.methods_and_results_synthesis", "--research_plan", str(input_file), "--methods_and_ref_file", str(soundness_dir / "snippet_references.json"), "--ref_and_paper_file", str(soundness_dir / "snippet_papers.json"), "--output_file", str(soundness_dir / "methods_analysis.json"), "--llm_engine_name", llm_model, "--cost_log_file", str(soundness_dir / "soundness_costs.jsonl")]
                if litellm_name:
                    cmd5.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd5, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Method analysis failed: {result.stderr}")
                    st.stop()
                st.success("Method analysis completed!")

                # Step 6: Synthesize review
                status_placeholder.text("Synthesizing method-level soundness review...")
                progress_bar.progress(95)
                st.info("Creating comprehensive soundness assessment...")
                review_file = soundness_dir / "meta_review.json"
                cmd6 = ["python", "-m", "ScholarEval.soundness.meta_review", "--research_plan", str(input_file), "--mr_analysis_file", str(soundness_dir / "methods_analysis.json"), "--methods_and_ref_file", str(soundness_dir / "snippet_references.json"), "--output_file", str(review_file), "--markdown_output", str(soundness_dir / "meta_review.md"), "--llm_engine_name", llm_model, "--cost_log_file", str(soundness_dir / "soundness_costs.jsonl")]
                if litellm_name:
                    cmd6.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd6, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Meta review failed: {result.stderr}")
                    st.stop()
                st.success("Meta review completed!")

                # Step 6: Synthesize review
                status_placeholder.text("Generating TL;DR")
                progress_bar.progress(95)
                st.info("Generating TLDR...")
                tldr_file = soundness_dir / "tldr_soundness.txt"
                cmd7 = ["python", "-m", "ScholarEval.soundness.tldr_soundness", "--input_file", str(input_file), "--meta_review_file", str(soundness_dir / "meta_review.json"), "--llm_engine_name", llm_model, "--output_file", str(tldr_file), 
                "--markdown_file",str(soundness_dir / "tldr_soundness.md"), "--cost_log_file", str(soundness_dir / "soundness_costs.jsonl")]
                if litellm_name:
                    cmd7.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd7, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"TLDR failed: {result.stderr}")
                    st.stop()
                st.success("TLDR completed!")

                meta_review_md = soundness_dir / "meta_review.md"
                # Load the meta review
                with open(meta_review_md, 'r') as f:
                    meta_review_content = f.read()

                # Load the TLDR
                with open(tldr_file, 'r') as f:
                    tldr_content = f.read()

                # Display review inside collapsible expander
                with st.expander("View Detailed Soundness Review", expanded=False):
                    st.markdown(meta_review_content)

                # Display TLDR fully
                st.markdown("### TL;DR Soundness Summary")
                st.markdown(tldr_content)

                end_time = time.time()
                total_time = end_time - start_time
                
                progress_bar.progress(100)
                status_placeholder.text("Soundness review complete!")
                
                # Show timing
                total_time_minutes = total_time / 60
                st.info(f"Soundness review completed in {total_time_minutes:.1f} minutes")
                
                # Download button
                st.download_button(
                    label="📥 Download Soundness Review",
                    data=meta_review_content,
                    file_name=f"soundness_review_{int(time.time())}.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Soundness review failed: {e}")
    with tabs[1]:
        st.subheader("Contribution Review")
        if st.button("Generate Contribution Review", disabled=not can_generate, key="contribution_btn"):
            st.info("Running contribution review pipeline...")
            
            try:
                start_time = time.time()
                
                # Create demo_data directory for persistent storage
                demo_data_path = ensure_demo_data_dir()
                contribution_dir = demo_data_path / f"contribution_{int(time.time())}"
                contribution_dir.mkdir(exist_ok=True)
                
                input_file = contribution_dir / "research_plan.txt"
                with open(input_file, 'w') as f:
                    f.write(research_plan_text)
                
                st.info(f"Saving intermediate results to: `{contribution_dir}`")
                
                # Set up environment - always run from root directory where packages are installed
                env = os.environ.copy()
                # Add current directory to Python path for local imports
                import sys
                working_dir = "."  # Always use root directory
                current_python_path = ":".join(sys.path)
                env["PYTHONPATH"] = f"{os.path.abspath('.')}:{current_python_path}:{env.get('PYTHONPATH', '')}"

                # Example pipeline steps
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                # Step 1: Extract dimensions and contributions
                status_placeholder.text("Extracting dimensions and contributions from research plan...")
                progress_bar.progress(10)
                st.info("Analyzing research plan to identify contribution dimensions...")
                cmd1 = ["python", "-m", "ScholarEval.contribution.extract_dimensions_and_contributions", "--input_file", str(input_file), "--llm_engine", "GPT-4.1-nano", "--output_file", str(contribution_dir / "dimensions_contributions.jsonl"), "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")]
                if litellm_name:
                    cmd1.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd1, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Dimension extraction failed: {result.stderr}")
                    st.stop()
                st.success("Dimensions and contributions extracted successfully!")
                
                # Display extracted dimensions and contributions
                try:
                    dimensions_list = []
                    with open(contribution_dir / "dimensions_contributions.jsonl", 'r') as f:
                            for line in f:
                                if line.strip():
                                    dimensions_list.append(json.loads(line))
                            display_raw_json(dimensions_list, "Extracted Dimensions and Contributions")
                except Exception as e:
                    st.warning(f"Could not display dimensions: {e}")

                # Step 2: Generate contribution queries
                status_placeholder.text("Generating targeted search queries...")
                progress_bar.progress(20)
                st.info("Creating optimized queries for literature search...")
                cmd2 = ["python", "-m", "ScholarEval.contribution.queries_generator", "--research_plan", str(input_file), "--contrib_file", str(contribution_dir / "dimensions_contributions.jsonl"), "--llm_engine_name", "GPT-4.1-nano", "--output_file", str(contribution_dir / "contribution_queries.json"), "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")]
                if litellm_name:
                    cmd2.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd2, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Query generation failed: {result.stderr}")
                    st.stop()
                st.success("Search queries generated successfully!")
                
                # Display generated queries
                try:
                    with open(contribution_dir / "contribution_queries.json", 'r') as f:
                        queries_data = json.load(f)
                    display_raw_json(queries_data, "Generated Contribution Queries")
                except Exception as e:
                    st.warning(f"Could not display queries: {e}")

                # Step 3: Extract papers using queries with progress monitoring
                status_placeholder.text("Extracting relevant papers from Semantic Scholar...")
                progress_bar.progress(25)
                st.info("Searching academic literature for relevant papers...")
                
                # Set up progress monitoring for paper extraction
                progress_file = contribution_dir / "paper_progress.json"
                cmd3 = ["python", "-m", "ScholarEval.contribution.paper_extractor", "--queries_file", str(contribution_dir / "contribution_queries.json"), "--output_file", str(contribution_dir / "contribution_papers.json"), "--progress_file", str(progress_file)]
                if cutoff_date:
                    cmd3.extend(["--cutoff_date", str(cutoff_date)])
                
                # Start the subprocess for paper extraction
                process = subprocess.Popen(cmd3, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=working_dir, env=env)
                
                # Monitor progress
                final_progress = monitor_progress_with_details(process, progress_file, status_placeholder, progress_bar, 25, 20)
                
                # Wait for completion and check result
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    st.error(f"Paper extraction failed: {stderr}")
                    st.stop()
                
                progress_bar.progress(45)
                st.success("Papers extracted successfully!")
                
                # Display paper summary
                try:
                    with open(contribution_dir / "contribution_papers.json", 'r') as f:
                        papers_data = json.load(f)
                    st.info(f"Found {len(papers_data)} papers from literature search")
                    display_raw_json(papers_data, "Extracted Papers")
                except Exception as e:
                    st.warning(f"Could not display papers: {e}")

                # Step 4: Assess relevance of extracted papers
                status_placeholder.text("Assessing relevance of extracted papers...")
                progress_bar.progress(50)
                st.info("Filtering papers based on relevance to research plan...")
                cmd4 = ["python", "-m", "ScholarEval.contribution.relevance_assessor", "--research_plan", str(input_file), "--papers_file", str(contribution_dir / "contribution_papers.json"), "--llm_engine", llm_model, "--output_file", str(contribution_dir / "filtered_contribution_papers.json"), "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")]
                if litellm_name:
                    cmd4.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd4, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Relevance assessment failed: {result.stderr}")
                    st.stop()
                st.success("Paper relevance assessed successfully!")
                
                # Display filtered papers summary
                try:
                    with open(contribution_dir / "filtered_contribution_papers.json", 'r') as f:
                        filtered_papers = json.load(f)["papers"]
                    # Filter papers with relevance_score > 3
                    original_count = len(filtered_papers)
                    high_relevance_papers = [paper for paper in filtered_papers if paper.get('relevance_score', 0) >= 3]
                    st.info(f"{len(high_relevance_papers)} relevant papers found")
                except Exception as e:
                    st.warning(f"Could not display filtered paper count: {e}")

                # Step 5: Augment papers with references and recommendations
                status_placeholder.text("Augmenting papers with references and recommendations...")
                progress_bar.progress(65)
                st.info("Expanding paper collection using references and recommendations...")
                cmd5 = ["python", "-m", "ScholarEval.contribution.paper_augmentation", "--relevant_papers", str(contribution_dir / "filtered_contribution_papers.json"), "--output_file", str(contribution_dir / "augmented_contribution_papers.json")]
                if cutoff_date:
                    cmd5.extend(["--cutoff_date", str(cutoff_date)])
                result = subprocess.run(cmd5, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Paper augmentation failed: {result.stderr}")
                    st.stop()
                st.success("Papers augmented successfully!")
                
                # Display augmented papers summary
                try:
                    with open(contribution_dir / "augmented_contribution_papers.json", 'r') as f:
                        augmented_papers = json.load(f)
                    st.info(f"Added {len(augmented_papers)-original_count} papers after augmentation")
                except Exception as e:
                    st.warning(f"Could not display augmented paper count: {e}")

                # Step 6: Embedding filter step 
                status_placeholder.text("Embedding-based filtering of augmented papers...")
                progress_bar.progress(70)
                st.info("Applying embedding-based filtering...")
                cmd6 = ["python", "-m", "ScholarEval.contribution.embedding_filter", "--research_plan", str(input_file), "--papers_json", str(contribution_dir / "augmented_contribution_papers.json"), "--output", str(contribution_dir / "filtered_augmented_contribution_papers.json"), "--top_k", "100"]
                result = subprocess.run(cmd6, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Embedding filter failed: {result.stderr}")
                    st.stop()
                st.success("Embedding filter successful!")

                # Step 7: Re-assess relevance of augmented papers
                status_placeholder.text("Re-assessing relevance of augmented papers...")
                progress_bar.progress(75)
                st.info("Filtering augmented papers for final relevance...")
                cmd7 = ["python", "-m", "ScholarEval.contribution.relevance_assessor", "--research_plan", str(input_file), "--papers_file", str(contribution_dir / "filtered_augmented_contribution_papers.json"), "--llm_engine", llm_model, "--output_file", str(contribution_dir / "final_contribution_papers.json"), "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")]
                if litellm_name:
                    cmd7.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd7, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Final relevance assessment failed: {result.stderr}")
                    st.stop()
                st.success("Final paper set determined!")
                
                # Display final papers summary
                try:
                    with open(contribution_dir / "final_contribution_papers.json", 'r') as f:
                        final_papers = json.load(f)["papers"]
                    high_relevance_final_papers = [paper for paper in final_papers if paper.get('relevance_score', 0) >= 3]
                    st.info(f"Final set: {len(high_relevance_final_papers)} papers for comparison analysis")
                except Exception as e:
                    st.warning(f"Could not display final paper count: {e}")

                # Step 8: Downsample papers for pairwise comparison
                status_placeholder.text("Sampling papers for final comparison...")
                progress_bar.progress(80)
                st.info("Downsampling papers for comparison...")
                cmd8 = ["python", "-m", "ScholarEval.contribution.paper_sampler", "--input_file", str(contribution_dir / "final_contribution_papers.json"), "--output_file", str(contribution_dir / "sampled_final_contribution_papers.json")]
                result = subprocess.run(cmd8, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Downsampling failed: {result.stderr}")
                    st.stop()
                st.success("Papers sampled successfully!")

                # Step 9: Conduct pairwise comparison
                status_placeholder.text("Conducting pairwise comparisons...")
                progress_bar.progress(85)
                st.info("Performing detailed comparisons between research plan and papers...")
                cmd9 = ["python", "-m", "ScholarEval.contribution.pairwise_comparator", "--research_plan", str(input_file), "--papers_metadata", str(contribution_dir / "sampled_final_contribution_papers.json"), "--dimensions_file", str(contribution_dir / "dimensions_contributions.jsonl"), "--llm_engine", llm_model, "--output_file", str(contribution_dir / "pairwise_comparisons.json"), "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")]
                if litellm_name:
                    cmd9.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd9, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Pairwise comparison failed: {result.stderr}")
                    st.stop()
                st.success("Pairwise comparisons completed!")

                # Step 10: Prepare final contribution context
                status_placeholder.text("Finalizing comparisons...")
                progress_bar.progress(90)
                st.info("Preparing context for final evaluation...")
                cmd10 = ["python", "-m", "ScholarEval.contribution.prepare_final_contribution_context", "--input_file", str(contribution_dir / "pairwise_comparisons.json"), "--output_file", str(contribution_dir / "contribution_context.json")]
                result = subprocess.run(cmd10, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Context preparation failed: {result.stderr}")
                    st.stop()
                st.success("Context preparation completed!")


                # Step 11: Generate final contribution review
                status_placeholder.text("Synthesizing final contribution review...")
                progress_bar.progress(95)
                st.info("Creating comprehensive contribution assessment...")
                review_file = contribution_dir / "contribution_review.txt"
                cmd11 = ["python", "-m", "ScholarEval.contribution.contribution_review_synthesis", "--research_plan", str(input_file), "--comparisons_file", str(contribution_dir / "contribution_context.json"), "--llm_engine", llm_model, "--output_file", str(review_file), "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")]
                if litellm_name:
                    cmd11.extend(["--litellm_name", litellm_name])
                result = subprocess.run(cmd11, capture_output=True, text=True, cwd=working_dir, env=env)
                if result.returncode != 0:
                    st.error(f"Contribution review synthesis failed: {result.stderr}")
                    st.stop()
                st.success("Contribution review completed!")

                # Show final result with timing
                with open(review_file, 'r') as f:
                    contribution_text = f.read()

                end_time = time.time()
                total_time = end_time - start_time
                
                progress_bar.progress(100)
                status_placeholder.text("Contribution review complete!")

                # Display structured review
                st.markdown("### Contribution Review Results")
                st.markdown(contribution_text)
                # Show timing
                total_time_minutes = total_time / 60
                st.info(f"Contribution review completed in {total_time_minutes:.1f} minutes")
                
                # Download button
                st.download_button(
                    label="Download Contribution Review",
                    data=contribution_text,
                    file_name=f"contribution_review_{int(time.time())}.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Contribution review failed: {e}")