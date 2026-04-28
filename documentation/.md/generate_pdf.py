"""
K-talysticFlow Documentation PDF Generator
==========================================
Combines all Wiki markdown files into a single professional PDF document.

Author: Késsia Souza Santos
Date: October 10, 2025
Version: 1.0.0
"""

import os
import re
from pathlib import Path
import subprocess
import sys

# Configuration
WIKI_DIR = Path(__file__).parent
OUTPUT_FILE = WIKI_DIR / "KAST-Complete-Documentation.pdf"
COMBINED_MD = WIKI_DIR / "KAST-Combined.md"

# Order of files for logical reading
FILE_ORDER = [
    "Installation.md",
    "User-Manual.md",
    "Pipeline-Steps.md",
    "Parallel-Processing.md",
    "Output-Analysis.md",
    "Configuration.md",
    "FAQ.md",
    "Troubleshooting.md",
]

def create_cover_page():
    """Create a professional cover page"""
    cover = """---
title: "K-talysticFlow"
subtitle: "Complete Documentation - Automated Deep Learning Pipeline for Molecular Bioactivity Prediction"
author: "Késsia Souza Santos"
date: "October 10, 2025"
version: "v1.0.0"
institute: "Laboratory of Molecular Modeling (LMM-UEFS)"
toc: true
toc-depth: 3
numbersections: true
geometry: margin=1in
fontsize: 11pt
colorlinks: true
linkcolor: blue
---

\\newpage

# About This Documentation

This comprehensive documentation covers all aspects of K-talysticFlow (KAST), an automated deep learning pipeline for molecular bioactivity prediction and virtual screening.

**Project Information:**

- **Name:** K-talysticFlow (KAST - K-atalystic Automated Screening Taskflow)
- **Version:** 1.0.0 (Stable Release)
- **Release Date:** October 10, 2025
- **Developer:** Késsia Souza Santos (@kelsouzs)
- **Institution:** Laboratory of Molecular Modeling, UEFS
- **Funding:** CNPq
- **License:** MIT License

**How to Use This Document:**

- **Beginners:** Start with sections 1-3 (Introduction, Installation, User Manual)
- **Regular Users:** Focus on sections 4-7 (Pipeline, Performance, Analysis, Configuration)
- **Advanced Users:** Review sections 8-10 (FAQ, Troubleshooting, Advanced Topics)
- **Reference:** Use section 11-12 (Quick Reference, Index)

\\newpage

"""
    return cover

def clean_markdown_for_pdf(content, filename):
    """Clean and prepare markdown content for PDF conversion"""
    
    # Add section title  
    section_title = f"\n\n# {filename.replace('.md', '').replace('-', ' ')}\n\n"
    
    # Remove HTML tags that don't work in PDF
    content = content.replace('<div align="center">', '')
    content = content.replace('</div>', '')
    content = content.replace('<p>', '')
    content = content.replace('</p>', '')
    content = content.replace('<em>', '*')
    content = content.replace('</em>', '*')
    content = content.replace('<strong>', '**')
    content = content.replace('</strong>', '**')
    content = content.replace('<h2>', '## ')
    content = content.replace('</h2>', '')
    
    # Remove badge images (not needed in PDF)
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        if '![' in line and 'badge' in line.lower():
            continue  # Skip badge lines
        if 'https://img.shields.io' in line:
            continue  # Skip shield badges
        cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    # Fix mermaid diagrams (Pandoc doesn't support them well)
    if '```mermaid' in content:
        # Replace mermaid blocks with text
        parts = content.split('```mermaid')
        result = parts[0]
        for part in parts[1:]:
            # Find the end of the code block
            if '```' in part:
                end_block = part.split('```', 1)[1]
                result += '```\n*[Diagram available in online documentation]*\n```' + end_block
            else:
                result += '```' + part
        content = result
    
    # Remove problematic Unicode characters that LaTeX can't handle
    # Replace em-dash with regular dash
    content = content.replace('—', '--')
    # Replace quotes
    content = content.replace('"', '"').replace('"', '"')
    content = content.replace(''', "'").replace(''', "'")
    
    # Remove ALL emojis and special Unicode symbols comprehensively
    # This is more aggressive - remove any character outside standard ASCII/Latin
    cleaned_chars = []
    for char in content:
        code = ord(char)
        # Keep ASCII (0-127) and Latin Extended (128-256)
        if code < 256:
            # For extended Latin, allow common accented characters
            if code >= 128:
                # Keep common accented characters (é, ñ, ü, etc.)
                if char in 'áéíóúàèìòùäëïöüñçÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÑÇ':
                    cleaned_chars.append(char)
                # Skip other extended Unicode in code blocks
                elif code in range(0x2000, 0x2700):  # General Punctuation, Currency, etc
                    cleaned_chars.append(' ')  # Replace with space
                else:
                    cleaned_chars.append(' ')
            else:
                cleaned_chars.append(char)
        elif char == '\n' or char == '\t':
            cleaned_chars.append(char)  # Keep newlines and tabs
        else:
            # Replace high Unicode with space (includes all emojis)
            cleaned_chars.append(' ')
    
    content = ''.join(cleaned_chars)
    
    # Remove standalone --- lines that conflict with YAML parsing
    # Replace with blank lines to maintain document structure
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        if line.strip() == '---':
            # Skip lines that are only ---
            fixed_lines.append('')
        else:
            fixed_lines.append(line)
    content = '\n'.join(fixed_lines)
    
    return section_title + content

def combine_markdown_files():
    """Combine all markdown files in the specified order"""
    print("📚 Combining markdown files...")
    
    combined_content = create_cover_page()
    
    for filename in FILE_ORDER:
        filepath = WIKI_DIR / filename
        if filepath.exists():
            print(f"  ✅ Adding: {filename}")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                cleaned_content = clean_markdown_for_pdf(content, filename)
                combined_content += cleaned_content + "\n\n"
        else:
            print(f"  ⚠️  Not found: {filename}")
    
    # Write combined markdown
    with open(COMBINED_MD, 'w', encoding='utf-8') as f:
        f.write(combined_content)
    
    print(f"\n✅ Combined markdown created: {COMBINED_MD}")
    return COMBINED_MD

def check_pandoc():
    """Check if Pandoc is installed"""
    try:
        result = subprocess.run(['pandoc', '--version'], 
                              capture_output=True, 
                              text=True,
                              encoding='utf-8')
        print(f"✅ Pandoc found: {result.stdout.split()[1]}")
        return True
    except FileNotFoundError:
        print("❌ Pandoc not found!")
        print("\nTo install Pandoc:")
        print("  Windows: choco install pandoc  OR  Download from https://pandoc.org")
        print("  Linux: sudo apt install pandoc")
        print("  Mac: brew install pandoc")
        return False

def generate_pdf():
    """Generate PDF using Pandoc"""
    if not check_pandoc():
        return False
    
    print("\n📄 Generating PDF...")
    
    # Pandoc command with professional options
    cmd = [
        'pandoc',
        str(COMBINED_MD),
        '-o', str(OUTPUT_FILE),
        '--pdf-engine=pdflatex',  # Use pdflatex for better compatibility
        '--toc',  # Table of contents
        '--toc-depth=3',
        '--number-sections',  # Number sections
        '--highlight-style=tango',  # Code highlighting
        '--from=markdown+raw_html',
        '--variable', 'geometry:margin=1in',
        '--variable', 'fontsize=11pt',
        '--variable', 'colorlinks=true',
        '--variable', 'linkcolor=blue',
        '--variable', 'urlcolor=blue',
        '--variable', 'toccolor=blue',
        '--metadata', 'title=K-talysticFlow Documentation',
    ]
    
    try:
        # Run with proper encoding and error handling
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            errors='replace',
            check=False  # Don't raise on non-zero exit code
        )
        
        if result.returncode == 0:
            if OUTPUT_FILE.exists():
                print(f"\n✅ PDF generated successfully: {OUTPUT_FILE}")
                print(f"📊 File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
                return True
            else:
                print(f"\n⚠️ PDF file not found after generation")
                return False
        else:
            print(f"\n❌ Pandoc error (code {result.returncode}):")
            if result.stderr:
                # Show only first 500 chars to avoid flooding output
                error_msg = result.stderr[:500]
                print(error_msg)
            print("\n💡 Trying alternative: converting without table of contents...")
            
            # Fallback: try without TOC
            cmd_fallback = [
                'pandoc',
                str(COMBINED_MD),
                '-o', str(OUTPUT_FILE),
                '--pdf-engine=pdflatex',
                '--number-sections',
                '--highlight-style=tango',
                '--from=markdown+raw_html',
            ]
            result2 = subprocess.run(
                cmd_fallback,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if result2.returncode == 0 and OUTPUT_FILE.exists():
                print(f"✅ PDF generated (without TOC): {OUTPUT_FILE}")
                print(f"📊 File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
                return True
            else:
                print("❌ Both attempts failed")
                return False
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

def main():
    """Main function"""
    print("="*60)
    print(" K-talysticFlow Documentation PDF Generator")
    print(" Version 1.0.0")
    print("="*60)
    print()
    
    # Step 1: Combine markdown files
    combined_md = combine_markdown_files()
    
    # Step 2: Generate PDF
    if generate_pdf():
        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print("="*60)
        print(f"\n📖 Your PDF is ready: {OUTPUT_FILE}")
        print(f"\n📂 Location: {OUTPUT_FILE.absolute()}")
        
        # Option to clean up combined markdown
        cleanup = input("\n🗑️  Delete temporary combined markdown? (y/n): ").lower()
        if cleanup == 'y':
            COMBINED_MD.unlink()
            print("✅ Cleaned up temporary files")
    else:
        print("\n" + "="*60)
        print("⚠️  PDF generation failed")
        print("="*60)
        print(f"\n📄 Combined markdown is available: {COMBINED_MD}")
        print("You can manually convert it using Pandoc or an online converter.")
    
    print("\n" + "="*60)
    print()

if __name__ == "__main__":
    main()
