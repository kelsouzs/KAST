#!/usr/bin/env python3
"""
K-talysticFlow (KAST) Wiki HTML Generator
Generates a single, beautiful HTML file from all Wiki markdown files with full emoji support
"""

import os
import re
from pathlib import Path
from datetime import datetime

try:
    import markdown2
except ImportError:
    print("Installing markdown2...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'markdown2'])
    import markdown2

# Configuration
WIKI_DIR = Path(__file__).parent
OUTPUT_FILE = WIKI_DIR / "KAST-Complete-Documentation.html"

# Order of Wiki pages
PAGE_ORDER = [
    "Home.md",
    "Installation.md",
    "User-Manual.md",
    "Pipeline-Steps.md",
    "Configuration.md",
    "Parallel-Processing.md",
    "Output-Analysis.md",
    "FAQ.md",
    "Troubleshooting.md"
]

def create_html_template():
    """Create HTML template with beautiful styling"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <meta name="description" content="K-talysticFlow - Automated Deep Learning Pipeline for Molecular Bioactivity Prediction">
    <meta name="theme-color" content="#667eea">
    <title>K-talysticFlow (KAST) - Complete Documentation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Segoe UI Emoji', 'Apple Color Emoji', 
                         'Noto Color Emoji', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
        }
        
        .container {
            max-width: 100%;
            width: 100%;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 30px rgba(0,0,0,0.1);
        }
        
        /* Cover Page */
        .cover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 100px 50px;
            text-align: center;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        
        .cover.hidden {
            display: none !important;
        }
        
        /* Hide cover if URL has hash on load (except #home) */
        body.has-hash .cover {
            display: none;
        }
        
        /* Show cover if hash is #home */
        body.home-hash .cover {
            display: flex !important;
        }
        
        .cover h1 {
            font-size: 3.5em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            animation: fadeInDown 1s ease-out;
        }
        
        .cover .subtitle {
            font-size: 1.5em;
            margin-bottom: 15px;
            opacity: 0.95;
            animation: fadeInUp 1s ease-out 0.3s both;
        }
        
        .cover .info {
            margin-top: 50px;
            font-size: 1.1em;
            opacity: 0.9;
            animation: fadeIn 1s ease-out 0.6s both;
        }
        
        .cover .badges {
            margin-top: 30px;
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .badge {
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 25px;
            backdrop-filter: blur(10px);
            font-weight: 500;
        }
        
        .scroll-indicator {
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            color: white;
            opacity: 0.9;
            z-index: 10;
        }
        
        .scroll-indicator p {
            margin: 5px 0;
            font-size: 1em;
            font-weight: 500;
        }
        
        .scroll-arrow {
            font-size: 2.5em;
            animation: bounce 2s infinite;
            display: block;
        }
        
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-10px);
            }
            60% {
                transform: translateY(-5px);
            }
        }
        
        /* Navigation Tabs */
        .nav {
            position: sticky;
            top: 0;
            background: #2c3e50;
            color: white;
            padding: 0;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .nav-header {
            padding: 10px 30px;
            background: #1a252f;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .nav-header h2 {
            font-size: 1em;
            margin: 0;
            color: white;
            font-weight: 600;
        }
        
        .nav-links {
            display: flex;
            flex-wrap: wrap;
            gap: 0;
            padding: 0 30px;
            background: #2c3e50;
            justify-content: flex-start;
        }
        
        .nav-links a {
            color: #ecf0f1;
            text-decoration: none;
            padding: 12px 24px;
            background: transparent;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            font-size: 0.9em;
            cursor: pointer;
            white-space: nowrap;
            flex-shrink: 0;
        }
        
        .nav-links a:hover {
            background: rgba(52, 152, 219, 0.1);
            border-bottom-color: #3498db;
        }
        
        .nav-links a.active {
            background: rgba(52, 152, 219, 0.2);
            border-bottom-color: #3498db;
            font-weight: 600;
        }
        
        /* Content */
        .content {
            padding: 30px 50px;
            min-height: calc(100vh - 200px);
        }
        
        .section {
            display: none;
            animation: fadeIn 0.4s ease-out;
        }
        
        .section.active {
            display: block;
        }
        
        .section-title {
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 4px solid #3498db;
            position: relative;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .section-title::before {
            content: '';
            position: absolute;
            bottom: -4px;
            left: 0;
            width: 100px;
            height: 4px;
            background: #e74c3c;
        }
        
        .page-indicator {
            position: fixed;
            top: 70px;
            right: 30px;
            background: rgba(52, 152, 219, 0.1);
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 0.9em;
            color: #2c3e50;
            backdrop-filter: blur(10px);
            z-index: 999;
            animation: fadeIn 0.3s ease-out;
        }
        
        h1 {
            font-size: 2.2em;
            color: #2c3e50;
            margin: 20px 0 20px 0;
            padding-left: 15px;
            border-left: 5px solid #3498db;
        }
        
        h1:first-child {
            margin-top: 0;
        }
        
        h2 {
            font-size: 1.8em;
            color: #34495e;
            margin: 35px 0 15px 0;
            padding-left: 12px;
            border-left: 4px solid #9b59b6;
        }
        
        h3 {
            font-size: 1.4em;
            color: #555;
            margin: 25px 0 12px 0;
            padding-left: 10px;
            border-left: 3px solid #1abc9c;
        }
        
        h4 {
            font-size: 1.2em;
            color: #666;
            margin: 20px 0 10px 0;
        }
        
        p {
            margin-bottom: 15px;
            text-align: justify;
        }
        
        /* Code blocks */
        code {
            background: #f4f4f4;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Courier New', Consolas, Monaco, monospace;
            font-size: 0.9em;
            color: #e74c3c;
            border: 1px solid #e0e0e0;
        }
        
        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        pre code {
            background: none;
            color: inherit;
            padding: 0;
            border: none;
            font-size: 0.95em;
            line-height: 1.5;
        }
        
        /* Lists */
        ul, ol {
            margin-left: 30px;
            margin-bottom: 20px;
        }
        
        li {
            margin-bottom: 10px;
            line-height: 1.7;
        }
        
        ul li::marker {
            color: #3498db;
            font-weight: bold;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }
        
        td {
            border: 1px solid #e0e0e0;
            padding: 12px 15px;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        tr:hover {
            background-color: #f0f7ff;
            transition: background-color 0.3s ease;
        }
        
        /* Blockquotes */
        blockquote {
            border-left: 5px solid #3498db;
            padding: 15px 20px;
            margin: 20px 0;
            background: #f8f9fa;
            border-radius: 0 8px 8px 0;
            font-style: italic;
            color: #555;
        }
        
        /* Links */
        a {
            color: #3498db;
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: all 0.3s ease;
        }
        
        a:hover {
            border-bottom-color: #3498db;
            color: #2980b9;
        }
        
        /* Images */
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin: 20px auto;
            display: block;
        }
        
        .diagram-container {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .diagram-container img {
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }
        
        /* Horizontal rule */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, transparent, #3498db, transparent);
            margin: 40px 0;
        }
        
        /* Special boxes */
        .info-box {
            background: #e8f4f8;
            border-left: 5px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .warning-box {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .success-box {
            background: #d4edda;
            border-left: 5px solid #28a745;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        
        /* Footer */
        .footer {
            background: #2c3e50;
            color: white;
            padding: 20px 50px;
            text-align: center;
        }
        
        .footer p {
            margin: 5px 0;
            text-align: center;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Scroll to top button */
        .scroll-top {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            opacity: 0;
            visibility: hidden;
            font-size: 1.5em;
        }
        
        .scroll-top.visible {
            opacity: 1;
            visibility: visible;
        }
        
        .scroll-top:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.3);
        }
        
        /* Print styles */
        @media print {
            .nav, .scroll-top {
                display: none;
            }
            
            .cover {
                page-break-after: always;
            }
            
            .section {
                page-break-inside: avoid;
            }
            
            body {
                background: white;
            }
        }
        
        /* Responsive - Tablet */
        @media (max-width: 1024px) {
            .content {
                padding: 20px 30px;
            }
            
            .nav-links a {
                padding: 12px 18px;
                font-size: 0.85em;
            }
            
            .cover h1 {
                font-size: 2.5em;
            }
            
            .cover .subtitle {
                font-size: 1.3em;
            }
        }
        
        /* Responsive - Mobile */
        @media (max-width: 768px) {
            .cover {
                padding: 60px 20px;
                min-height: 100vh;
            }
            
            .cover h1 {
                font-size: 2em;
                line-height: 1.2;
            }
            
            .cover .subtitle {
                font-size: 1em;
                margin-bottom: 10px;
            }
            
            .cover .info {
                font-size: 0.9em;
                margin-top: 30px;
            }
            
            .cover .badges {
                flex-direction: column;
                gap: 10px;
            }
            
            .badge {
                padding: 8px 16px;
                font-size: 0.9em;
            }
            
            .nav-header {
                padding: 10px 20px;
                flex-direction: row;
                align-items: center;
                gap: 10px;
                flex-wrap: nowrap;
            }
            
            .nav-header h2 {
                font-size: 0.85em;
                white-space: nowrap;
            }
            
            .nav-header span {
                font-size: 0.8em !important;
            }
            
            .nav-links {
                padding: 0 15px;
                gap: 0;
                overflow-x: auto;
                flex-wrap: nowrap;
                -webkit-overflow-scrolling: touch;
            }
            
            .nav-links a {
                padding: 10px 16px;
                font-size: 0.8em;
                flex-shrink: 0;
            }
            
            .content {
                padding: 20px 15px;
            }
            
            .section-title {
                font-size: 1.8em;
            }
            
            h1 {
                font-size: 1.6em;
                margin: 15px 0 15px 0;
            }
            
            h2 {
                font-size: 1.3em;
            }
            
            h3 {
                font-size: 1.1em;
            }
            
            .footer {
                padding: 15px 20px;
                font-size: 0.9em;
            }
            
            .scroll-indicator {
                bottom: 100px;
                font-size: 0.9em;
            }
            
            .scroll-indicator p {
                font-size: 0.9em;
            }
            
            .scroll-arrow {
                font-size: 2em;
            }
            
            .scroll-top {
                bottom: 20px;
                right: 20px;
                width: 45px;
                height: 45px;
                font-size: 1.3em;
            }
            
            table {
                font-size: 0.85em;
                display: block;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
            }
            
            pre {
                font-size: 0.8em;
                padding: 15px;
                overflow-x: auto;
            }
            
            code {
                font-size: 0.85em;
            }
            
            .diagram-container {
                padding: 10px;
                margin: 15px 0;
            }
            
            blockquote {
                padding: 10px 15px;
                font-size: 0.95em;
            }
        }
        
        /* Responsive - Small Mobile */
        @media (max-width: 480px) {
            .cover h1 {
                font-size: 1.5em;
            }
            
            .cover .subtitle {
                font-size: 0.9em;
            }
            
            .nav-header h2 {
                font-size: 0.85em;
            }
            
            .nav-links a {
                padding: 10px 12px;
                font-size: 0.75em;
            }
            
            h1 {
                font-size: 1.4em;
            }
            
            h2 {
                font-size: 1.2em;
            }
            
            .content {
                padding: 15px 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Cover Page -->
        <div class="cover">
            <h1>🧬 K-talysticFlow (KAST)</h1>
            <div class="subtitle">📚 Complete Documentation & User Guide</div>
            <div class="subtitle">🔬 Automated Deep Learning Pipeline for Molecular Bioactivity Prediction</div>
            <div class="info">
                <strong>Version:</strong> 1.0.0<br>
                <strong>Developer:</strong> Késsia Souza Santos<br>
                <strong>Generated:</strong> {date}
            </div>
            <div class="badges">
                <span class="badge">⚡ High-Performance</span>
                <span class="badge">🎯 Accurate</span>
                <span class="badge">🚀 Automated</span>
            </div>
            <div class="scroll-indicator">
                <p>Scroll down to start</p>
                <div class="scroll-arrow">↓</div>
            </div>
        </div>
        
        <!-- Navigation -->
        <div class="nav">
            <div class="nav-header">
                <h2>🧬 K-talysticFlow</h2>
                <span style="opacity: 0.7; font-size: 0.9em;">Documentation</span>
            </div>
            <div class="nav-links">
{nav_links}
            </div>
        </div>
        
        <!-- Content -->
        <div class="content">
{content}
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p><strong>🧬 K-talysticFlow (KAST)</strong></p>
            <p>Automated Deep Learning Pipeline for Molecular Bioactivity Prediction</p>
            <hr style="border: none; height: 1px; background: rgba(255,255,255,0.2); margin: 15px 0;">
            <p>
                <strong>📞 Support:</strong> <a href="https://github.com/kelsouzs/KAST/issues" style="color: #3498db;">Create Issue</a> | 
                <strong>📧 Contact:</strong> <a href="mailto:lmm@uefs.br" style="color: #3498db;">lmm@uefs.br</a>
            </p>
            <p style="margin-top: 10px;">
                <a href="https://www.linkedin.com/in/kelsouzs" target="_blank" style="color: #0077B5; text-decoration: none; margin-right: 15px;">
                    <strong>💼 LinkedIn</strong>
                </a>
                <a href="https://orcid.org/0009-0001-5157-6638" target="_blank" style="color: #A6CE39; text-decoration: none;">
                    <strong>🆔 ORCID</strong>
                </a>
            </p>
            <p style="font-size: 0.9em; opacity: 0.8; margin-top: 10px;">© 2025 Késsia Souza Santos - K-talysticFlow is licensed under MIT | Last updated: October 2025</p>
            <p style="margin-top: 5px; font-size: 0.95em;"><em>Made with ❤️ for the computational chemistry community by Késsia Souza (@kelsouzs)</em></p>
        </div>
    </div>
    
    <!-- Scroll to top button -->
    <div class="scroll-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">
        ↑
    </div>
    
    <script>
        // Tab navigation system - SIMPLIFIED VERSION
        const navLinks = document.querySelectorAll('.nav-links a');
        const sections = document.querySelectorAll('.section');
        const coverPage = document.querySelector('.cover');
        
        // Function to show a specific section or home
        function showTab(targetSection) {
            console.log('📍 Showing tab:', targetSection);
            console.log('Cover page element:', coverPage);
            console.log('Total sections:', sections.length);
            
            // Remove active from all links and sections
            navLinks.forEach(l => l.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            
            // Find and activate the target link
            const targetLink = document.querySelector(`a[data-section="${targetSection}"]`);
            if (targetLink) {
                targetLink.classList.add('active');
                console.log('✅ Link activated:', targetSection);
            } else {
                console.error('❌ Link not found for:', targetSection);
            }
            
            if (targetSection === 'home') {
                // Show BOTH cover page AND home section
                if (coverPage) {
                    coverPage.classList.remove('hidden');
                    coverPage.style.display = 'flex'; // Force show
                    console.log('✅ Cover page shown');
                } else {
                    console.error('❌ Cover page element not found!');
                }
                
                // Also show home section content
                const homeSection = document.getElementById('home');
                if (homeSection) {
                    homeSection.classList.add('active');
                    console.log('✅ Home section shown');
                } else {
                    console.error('❌ Home section not found!');
                }
            } else {
                // Hide cover page, show only target section
                if (coverPage) {
                    coverPage.classList.add('hidden');
                    coverPage.style.display = 'none'; // Force hide
                    console.log('✅ Cover page hidden');
                }
                
                // Show target section
                const target = document.getElementById(targetSection);
                if (target) {
                    target.classList.add('active');
                    console.log('✅ Section activated:', targetSection);
                } else {
                    console.error('❌ Section not found:', targetSection);
                }
            }
            
            // Scroll to top
            window.scrollTo({top: 0, behavior: 'instant'});
        }
        
        // Click handlers for tabs
        navLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const targetSection = this.getAttribute('data-section');
                showTab(targetSection);
                // Update URL
                history.pushState(null, null, '#' + targetSection);
            });
        });
        
        // Handle browser back/forward buttons
        window.addEventListener('popstate', function() {
            const hash = window.location.hash.substring(1) || 'home';
            showTab(hash);
        });
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            const hash = window.location.hash.substring(1);
            console.log('🔄 Page loaded with hash:', hash || '(none)');
            
            if (hash && hash !== 'home') {
                // Load specific section from URL
                document.body.classList.add('has-hash');
                showTab(hash);
            } else {
                // Default to home (no hash or #home)
                showTab('home');
            }
        });
        
        // Show/hide scroll to top button
        window.addEventListener('scroll', function() {
            const scrollTop = document.querySelector('.scroll-top');
            if (window.pageYOffset > 300) {
                scrollTop.classList.add('visible');
            } else {
                scrollTop.classList.remove('visible');
            }
        });
        
        // Smooth scroll for internal links within content
        document.addEventListener('click', function(e) {
            if (e.target.tagName === 'A' && e.target.hash && e.target.hash.startsWith('#')) {
                const targetId = e.target.hash.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement && !e.target.hasAttribute('data-section')) {
                    e.preventDefault();
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    </script>
</body>
</html>
"""

def convert_image_to_base64(image_path):
    """Convert image file to base64 data URI"""
    import base64
    try:
        with open(image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            ext = image_path.suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.svg': 'image/svg+xml'
            }.get(ext, 'image/png')
            return f"data:{mime_type};base64,{img_data}"
    except Exception as e:
        print(f"    ⚠️  Warning: Could not convert image {image_path.name}: {e}")
        return None

def clean_markdown_content(content):
    """Clean markdown content by removing unwanted sections"""
    # Update all dates to current date
    content = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+202\d\b', 
                     'October 10, 2025', content, flags=re.IGNORECASE)
    content = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+202\d\b', 
                     'October 2025', content, flags=re.IGNORECASE)
    
    lines = content.split('\n')
    cleaned_lines = []
    skip_section = False
    skip_until_next_header = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Remove "Back to" links
        if re.match(r'.*\[.*Back to.*\]', line, re.IGNORECASE):
            i += 1
            continue
        
        # Remove "Return to" links
        if re.match(r'.*\[.*Return to.*\]', line, re.IGNORECASE):
            i += 1
            continue
        
        # Remove "Next:" links
        if re.match(r'.*\[.*Next:.*\]', line, re.IGNORECASE) or re.match(r'.*Next\s*:', line, re.IGNORECASE):
            i += 1
            continue
        
        # Remove KAST Wiki title and KAST emoji titles
        if re.match(r'^#\s*(💻\s*)?KAST(\s*🧪|\s+Wiki)?$', line.strip(), re.IGNORECASE):
            i += 1
            continue
            
        # Detect MacOS installation section
        if re.match(r'#{1,4}\s+.*macOS', line, re.IGNORECASE):
            skip_until_next_header = True
            i += 1
            continue
        
        # Detect Docker installation section
        if re.match(r'#{1,4}\s+.*Docker', line, re.IGNORECASE):
            skip_until_next_header = True
            i += 1
            continue
        
        # If we're skipping and hit a same-level or higher-level header, stop skipping
        if skip_until_next_header and re.match(r'^#{1,3}\s+', line):
            skip_until_next_header = False
        
        # Skip lines in unwanted sections
        if skip_until_next_header:
            i += 1
            continue
        
        # Add hyperlink to email
        line = re.sub(r'\blmm@uefs\.br\b', r'[lmm@uefs.br](mailto:lmm@uefs.br)', line)
        
        # Remove "Last Updated" lines
        if re.match(r'^\*?\*?Last Updated:.*202\d.*\*?\*?$', line.strip(), re.IGNORECASE):
            i += 1
            continue
        
        # Remove "Made with ❤️" lines (will be in footer instead)
        if re.match(r'.*Made with.*❤️.*computational chemistry.*', line, re.IGNORECASE):
            i += 1
            continue
        
        cleaned_lines.append(line)
        i += 1
    
    return '\n'.join(cleaned_lines)

def convert_markdown_to_html(markdown_file):
    """Convert markdown file to HTML"""
    print(f"  📄 Processing: {markdown_file.name}")
    
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean the content
    content = clean_markdown_content(content)
    
    # Replace local image paths with base64 data URIs
    def replace_image(match):
        img_text = match.group(1)
        img_path_with_attrs = match.group(2)
        
        # Remove any attributes like {width=80%}
        img_path = img_path_with_attrs.split('{')[0].split(')')[0].strip()
        
        # Skip external URLs
        if img_path.startswith('http://') or img_path.startswith('https://'):
            return match.group(0)
        
        # Try to find the image file
        possible_paths = [
            WIKI_DIR / img_path,
            WIKI_DIR / '.md' / img_path,
            WIKI_DIR.parent / img_path
        ]
        
        for path in possible_paths:
            if path.exists():
                base64_data = convert_image_to_base64(path)
                if base64_data:
                    print(f"    🖼️  Embedded image: {img_path}")
                    return f'![{img_text}]({base64_data})'
                break
        
        print(f"    ⚠️  Image not found: {img_path}")
        return match.group(0)
    
    # Replace image references (including those with attributes)
    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)(\{[^}]*\})?', replace_image, content)
    
    # Convert markdown to HTML with extras (removed 'header-ids' to avoid duplicate IDs)
    html = markdown2.markdown(content, extras=[
        'fenced-code-blocks',
        'tables',
        'break-on-newline',
        'footnotes',
        'smarty-pants',
        'strike',
        'task_list',
        'code-friendly'
    ])
    
    # Wrap images in diagram containers for better display
    html = re.sub(
        r'<img\s+([^>]*src=["\']([^"\']+)["\'][^>]*)>',
        r'<div class="diagram-container"><img \1></div>',
        html
    )
    
    return html

def generate_html():
    """Generate complete HTML documentation"""
    print("=" * 60)
    print("🧬 K-talysticFlow Wiki HTML Generator")
    print("=" * 60)
    print("\n🔄 Processing Wiki pages...")
    
    # Define emojis for each section
    section_emojis = {
        'home': '🏠',
        'installation': '⚙️',
        'user-manual': '📖',
        'pipeline-steps': '🔄',
        'configuration': '🔧',
        'parallel-processing': '⚡',
        'output-analysis': '📊',
        'faq': '❓',
        'troubleshooting': '🔧'
    }
    
    # Collect content
    sections = []
    nav_links = []
    
    for page in PAGE_ORDER:
        # Try multiple locations - prefer .md folder which has images
        page_paths = [
            WIKI_DIR / '.md' / page,
            WIKI_DIR / page
        ]
        
        page_path = None
        for path in page_paths:
            if path.exists():
                page_path = path
                print(f"  📂 Using: {path.relative_to(WIKI_DIR)}")
                break
        
        if page_path:
            section_name = page_path.stem.replace('-', ' ')
            section_id = page_path.stem.lower()
            emoji = section_emojis.get(section_id, '📄')
            
            # Convert markdown to HTML
            html_content = convert_markdown_to_html(page_path)
            
            # Wrap in section - no section active by default
            section_html = f'''
            <div class="section" id="{section_id}">
                {html_content}
            </div>
            '''
            
            sections.append(section_html)
            # Don't make any tab active by default - let JS handle it
            nav_links.append(f'<a href="#" data-section="{section_id}">{emoji} {section_name}</a>')
        else:
            print(f"  ⚠️  Warning: {page} not found, skipping...")
    
    # Build final HTML
    content_html = '\n'.join(sections)
    nav_html = '\n                '.join(nav_links)
    current_date = datetime.now().strftime('%B %d, %Y')
    
    final_html = create_html_template()
    final_html = final_html.replace('{date}', current_date)
    final_html = final_html.replace('{nav_links}', nav_html)
    final_html = final_html.replace('{content}', content_html)
    
    # Write to file
    print(f"\n📝 Generating HTML: {OUTPUT_FILE.name}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    # Get file size
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    
    print(f"\n✅ HTML generated successfully!")
    print(f"📦 File: {OUTPUT_FILE}")
    print(f"📏 Size: {size_kb:.2f} KB")
    print(f"📄 Sections: Combined {len(sections)} Wiki pages")
    print(f"\n🎉 HTML generation complete with full emoji support! 😊")
    print(f"📍 Location: {OUTPUT_FILE.absolute()}")
    print(f"\n💡 Tip: Open the HTML file in any browser to view the documentation!")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    generate_html()
