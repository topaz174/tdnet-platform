#!/usr/bin/env python3
"""
Simplified Parallel PDF Extraction Pipeline

A streamlined version of the parallel PDF extraction pipeline that focuses on
core functionality without optional dependencies.

This version:
- Works with basic Python libraries
- Provides significant speedup through multiprocessing
- Has minimal dependencies 
- Graceful fallbacks for missing libraries

Usage:
    python parallel_pdf_simple.py --pdf-dir /path/to/pdfs --workers 8
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import time
import concurrent.futures
import multiprocessing as mp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Suppress warnings during import
import warnings
warnings.filterwarnings("ignore")

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required dependencies
REQUIRED_AVAILABLE = True
try:
    # Import basic dependencies that should be available
    from pdf_extraction_pipeline import PDFProcessor, PDFExtractionTester
    logger.info("Successfully imported PDF processing components")
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    REQUIRED_AVAILABLE = False

@dataclass
class SimpleConfig:
    """Simple configuration for parallel processing"""
    max_workers: int = 8
    max_concurrent_files: int = 4
    output_dir: str = "parallel_output"
    disable_ocr: bool = True  # Disable OCR by default for speed
    debug: bool = False

class SimplePDFProcessor:
    """Simple wrapper around the original PDF processor"""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        
    def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file"""
        try:
            logger.info(f"Processing: {os.path.basename(pdf_path)}")
            
            # Use original processor
            processor = PDFProcessor(
                ocr_backend="none" if self.config.disable_ocr else "auto",
                disable_ocr=self.config.disable_ocr,
                smart_ocr=not self.config.disable_ocr
            )
            
            chunks = processor.process_pdf_file(pdf_path)
            
            result = {
                'filename': os.path.basename(pdf_path),
                'success': True,
                'chunks_count': len(chunks),
                'chunks': chunks
            }
            
            logger.info(f"Completed: {os.path.basename(pdf_path)} - {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {
                'filename': os.path.basename(pdf_path),
                'success': False,
                'error': str(e),
                'chunks_count': 0,
                'chunks': []
            }

def process_pdf_worker(args):
    """Worker function for multiprocessing"""
    pdf_path, config_dict = args
    
    # Recreate config object
    config = SimpleConfig(**config_dict)
    processor = SimplePDFProcessor(config)
    
    return processor.process_single_pdf(pdf_path)

class SimpleParallelPipeline:
    """Simple parallel processing pipeline using multiprocessing"""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Adjust worker count if needed
        cpu_count = mp.cpu_count()
        if config.max_workers > cpu_count:
            logger.warning(f"Requested {config.max_workers} workers, but only {cpu_count} CPUs available")
            self.config.max_workers = min(config.max_workers, cpu_count)
        
        logger.info(f"Initialized pipeline with {self.config.max_workers} workers")
    
    def process_files(self, pdf_files: List[str]) -> Dict[str, Any]:
        """Process multiple PDF files in parallel"""
        
        if not pdf_files:
            logger.error("No PDF files provided")
            return {}
        
        logger.info(f"Starting parallel processing of {len(pdf_files)} files...")
        start_time = time.time()
        
        # Prepare arguments for workers
        worker_args = [(pdf_path, asdict(self.config)) for pdf_path in pdf_files]
        
        # Process files with multiprocessing
        results = []
        failed_files = []
        total_chunks = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(process_pdf_worker, args): args[0] 
                for args in worker_args
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                pdf_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        total_chunks += result['chunks_count']
                        # Save individual file results
                        self._save_file_result(result)
                    else:
                        failed_files.append({
                            'filename': result['filename'],
                            'error': result.get('error', 'Unknown error')
                        })
                        
                except Exception as e:
                    logger.error(f"Worker failed for {pdf_file}: {e}")
                    failed_files.append({
                        'filename': os.path.basename(pdf_file),
                        'error': str(e)
                    })
        
        # Calculate statistics
        total_time = time.time() - start_time
        processed_files = len([r for r in results if r['success']])
        
        summary = {
            'total_files': len(pdf_files),
            'processed_files': processed_files,
            'failed_files': len(failed_files),
            'total_chunks': total_chunks,
            'total_time': total_time,
            'files_per_second': processed_files / total_time if total_time > 0 else 0,
            'chunks_per_second': total_chunks / total_time if total_time > 0 else 0,
            'failed_file_list': failed_files,
            'worker_count': self.config.max_workers
        }
        
        # Save summary
        self._save_summary(summary)
        
        logger.info(f"Processing completed in {total_time:.2f}s")
        logger.info(f"Processed: {processed_files}/{len(pdf_files)} files")
        logger.info(f"Total chunks: {total_chunks}")
        logger.info(f"Performance: {summary['files_per_second']:.2f} files/s")
        
        return summary
    
    def _save_file_result(self, result: Dict[str, Any]):
        """Save individual file processing result"""
        try:
            filename = result['filename']
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(self.config.output_dir, f"{base_name}_chunks.json")
            
            # Convert chunks to JSON-serializable format
            chunks_data = []
            for chunk in result['chunks']:
                if hasattr(chunk, '__dict__'):
                    chunks_data.append(asdict(chunk))
                else:
                    chunks_data.append(chunk)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving result for {result['filename']}: {e}")
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save processing summary"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(self.config.output_dir, f"processing_summary_{timestamp}.json")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")

def find_pdf_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """Find PDF files in directory"""
    pdf_files = []
    
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return []
    
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    
    if max_files:
        pdf_files = pdf_files[:max_files]
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files

def main():
    parser = argparse.ArgumentParser(description='Simple Parallel PDF Extraction Pipeline')
    parser.add_argument('--pdf-file', type=str, help='Path to a single PDF file')
    parser.add_argument('--pdf-dir', type=str, help='Directory containing PDF files')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes')
    parser.add_argument('--output-dir', type=str, default='simple_parallel_output', help='Output directory')
    parser.add_argument('--enable-ocr', action='store_true', help='Enable OCR processing (slower)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not REQUIRED_AVAILABLE:
        logger.error("Missing required dependencies. Please ensure pdf_extraction_pipeline.py is available.")
        sys.exit(1)
    
    if not args.pdf_file and not args.pdf_dir:
        print("Error: Please specify either --pdf-file or --pdf-dir")
        parser.print_help()
        sys.exit(1)
    
    # Auto-detect optimal worker count
    if args.workers == 8:  # default
        cpu_count = mp.cpu_count()
        optimal_workers = min(max(cpu_count // 2, 4), 16)  # Use 50% of cores, min 4, max 16
        args.workers = optimal_workers
        logger.info(f"Auto-detected optimal worker count: {args.workers}")
    
    # Create configuration
    config = SimpleConfig(
        max_workers=args.workers,
        output_dir=args.output_dir,
        disable_ocr=not args.enable_ocr,  # Disable OCR by default for speed
        debug=args.debug
    )
    
    # Get file list
    if args.pdf_file:
        pdf_files = [args.pdf_file]
    else:
        pdf_files = find_pdf_files(args.pdf_dir, args.max_files)
    
    if not pdf_files:
        logger.error("No PDF files found to process")
        sys.exit(1)
    
    # Process files
    pipeline = SimpleParallelPipeline(config)
    
    try:
        summary = pipeline.process_files(pdf_files)
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Files processed: {summary['processed_files']}/{summary['total_files']}")
        print(f"Total chunks: {summary['total_chunks']}")
        print(f"Processing time: {summary['total_time']:.2f}s")
        print(f"Throughput: {summary['files_per_second']:.2f} files/s")
        print(f"Workers used: {summary['worker_count']}")
        print(f"Results saved to: {config.output_dir}")
        
        if summary['failed_files'] > 0:
            print(f"\nFailed files: {summary['failed_files']}")
            for failed in summary['failed_file_list'][:5]:  # Show first 5 failures
                print(f"  - {failed['filename']}: {failed['error']}")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()