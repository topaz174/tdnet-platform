#!/usr/bin/env python3
"""
Benchmark Script for Parallel PDF Extraction Pipeline

This script compares the performance of the original PDF extraction pipeline
vs the new parallel implementation, providing detailed metrics and analysis.

Usage:
    python benchmark_parallel_pipeline.py --pdf-dir /path/to/test/pdfs --max-files 10
    python benchmark_parallel_pipeline.py --generate-test-data --test-files 5
"""

import os
import sys
import time
import argparse
import asyncio
import logging
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# Disable warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    pipeline_name: str
    total_files: int
    processed_files: int
    failed_files: int
    total_chunks: int
    total_time: float
    avg_time_per_file: float
    files_per_second: float
    chunks_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    errors: List[str]

@dataclass
class ComparisonReport:
    """Comparison report between pipelines"""
    original_result: BenchmarkResult
    parallel_result: BenchmarkResult
    speedup_factor: float
    throughput_improvement: float
    efficiency_gain: float
    memory_difference_mb: float

class PerformanceMonitor:
    """Monitor system performance during processing"""
    
    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            self.monitoring_enabled = True
        except ImportError:
            logger.warning("psutil not available - performance monitoring disabled")
            self.monitoring_enabled = False
            self.psutil = None
        
        self.start_time = None
        self.peak_memory = 0
        self.avg_cpu = 0
        self.measurements = []
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.monitoring_enabled:
            return
            
        self.start_time = time.time()
        self.peak_memory = 0
        self.measurements = []
        
        # Start background monitoring
        self.monitoring = True
        asyncio.create_task(self._monitor_loop())
    
    def stop_monitoring(self) -> Tuple[float, float]:
        """Stop monitoring and return (peak_memory_mb, avg_cpu_percent)"""
        if not self.monitoring_enabled:
            return 0.0, 0.0
            
        self.monitoring = False
        
        if self.measurements:
            avg_cpu = statistics.mean([m['cpu'] for m in self.measurements])
            peak_memory = max([m['memory'] for m in self.measurements])
        else:
            avg_cpu = 0.0
            peak_memory = 0.0
            
        return peak_memory, avg_cpu
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while getattr(self, 'monitoring', True):
            try:
                process = self.psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                self.measurements.append({
                    'memory': memory_mb,
                    'cpu': cpu_percent,
                    'timestamp': time.time()
                })
                
                await asyncio.sleep(1)  # Sample every second
                
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                break

class OriginalPipelineBenchmark:
    """Benchmark wrapper for the original PDF pipeline"""
    
    def __init__(self, output_dir: str = "benchmark_original"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Import original pipeline
        try:
            from pdf_extraction_pipeline import PDFExtractionTester
            self.tester_class = PDFExtractionTester
        except ImportError as e:
            logger.error(f"Could not import original pipeline: {e}")
            logger.info("Make sure pdf_extraction_pipeline.py is in the same directory")
            sys.exit(1)
    
    def run_benchmark(self, pdf_files: List[str]) -> BenchmarkResult:
        """Run benchmark on original pipeline"""
        logger.info(f"Benchmarking original pipeline with {len(pdf_files)} files")
        
        # Initialize performance monitoring
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        errors = []
        total_chunks = 0
        processed_files = 0
        
        try:
            tester = self.tester_class(self.output_dir)
            
            for pdf_file in pdf_files:
                try:
                    logger.info(f"Processing: {os.path.basename(pdf_file)}")
                    stats = tester.test_single_pdf(pdf_file)
                    
                    if 'error' in stats:
                        errors.append(f"{os.path.basename(pdf_file)}: {stats['error']}")
                    else:
                        total_chunks += stats.get('total_chunks', 0)
                        processed_files += 1
                        
                except Exception as e:
                    error_msg = f"{os.path.basename(pdf_file)}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
        except Exception as e:
            logger.error(f"Original pipeline benchmark failed: {e}")
            errors.append(f"Pipeline error: {str(e)}")
        
        total_time = time.time() - start_time
        peak_memory, avg_cpu = monitor.stop_monitoring()
        
        # Calculate metrics
        failed_files = len(pdf_files) - processed_files
        avg_time_per_file = total_time / len(pdf_files) if pdf_files else 0
        files_per_second = processed_files / total_time if total_time > 0 else 0
        chunks_per_second = total_chunks / total_time if total_time > 0 else 0
        
        result = BenchmarkResult(
            pipeline_name="Original",
            total_files=len(pdf_files),
            processed_files=processed_files,
            failed_files=failed_files,
            total_chunks=total_chunks,
            total_time=total_time,
            avg_time_per_file=avg_time_per_file,
            files_per_second=files_per_second,
            chunks_per_second=chunks_per_second,
            memory_usage_mb=peak_memory,
            cpu_usage_percent=avg_cpu,
            errors=errors
        )
        
        logger.info(f"Original pipeline completed in {total_time:.2f}s")
        return result

class ParallelPipelineBenchmark:
    """Benchmark wrapper for the parallel PDF pipeline"""
    
    def __init__(self, output_dir: str = "benchmark_parallel", workers: int = 16):
        self.output_dir = output_dir
        self.workers = workers
        os.makedirs(output_dir, exist_ok=True)
        
        # Import parallel pipeline
        try:
            from parallel_pdf_extraction_pipeline import (
                ParallelPDFExtractionPipeline, 
                ParallelProcessingConfig
            )
            self.pipeline_class = ParallelPDFExtractionPipeline
            self.config_class = ParallelProcessingConfig
        except ImportError as e:
            logger.error(f"Could not import parallel pipeline: {e}")
            sys.exit(1)
    
    async def run_benchmark(self, pdf_files: List[str]) -> BenchmarkResult:
        """Run benchmark on parallel pipeline"""
        logger.info(f"Benchmarking parallel pipeline with {len(pdf_files)} files, {self.workers} workers")
        
        # Initialize performance monitoring
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        errors = []
        
        try:
            # Configure parallel pipeline
            config = self.config_class(
                max_workers=self.workers,
                max_concurrent_files=min(8, len(pdf_files)),
                batch_size=4,
                use_gpu_ocr=False,  # Disable for fair comparison
                enable_progress_monitoring=False  # Reduce overhead
            )
            
            pipeline = self.pipeline_class(config, self.output_dir)
            summary = await pipeline.process_files_async(pdf_files)
            
            # Extract metrics from summary
            stats = summary.get('processing_stats', {})
            processed_files = stats.get('processed_files', 0)
            failed_files = stats.get('failed_files', 0)
            total_chunks = stats.get('total_chunks', 0)
            
            # Collect errors
            for failed in summary.get('failed_files', []):
                errors.append(f"{failed['filename']}: {failed['error']}")
            
        except Exception as e:
            logger.error(f"Parallel pipeline benchmark failed: {e}")
            errors.append(f"Pipeline error: {str(e)}")
            processed_files = 0
            failed_files = len(pdf_files)
            total_chunks = 0
        
        total_time = time.time() - start_time
        peak_memory, avg_cpu = monitor.stop_monitoring()
        
        # Calculate metrics
        avg_time_per_file = total_time / len(pdf_files) if pdf_files else 0
        files_per_second = processed_files / total_time if total_time > 0 else 0
        chunks_per_second = total_chunks / total_time if total_time > 0 else 0
        
        result = BenchmarkResult(
            pipeline_name="Parallel",
            total_files=len(pdf_files),
            processed_files=processed_files,
            failed_files=failed_files,
            total_chunks=total_chunks,
            total_time=total_time,
            avg_time_per_file=avg_time_per_file,
            files_per_second=files_per_second,
            chunks_per_second=chunks_per_second,
            memory_usage_mb=peak_memory,
            cpu_usage_percent=avg_cpu,
            errors=errors
        )
        
        logger.info(f"Parallel pipeline completed in {total_time:.2f}s")
        return result

class BenchmarkRunner:
    """Main benchmark runner"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    async def run_comparison_benchmark(self, pdf_files: List[str], workers: int = 16) -> ComparisonReport:
        """Run both pipelines and compare results"""
        logger.info(f"Starting benchmark comparison with {len(pdf_files)} files")
        
        # Run original pipeline
        original_benchmark = OriginalPipelineBenchmark(
            os.path.join(self.output_dir, "original")
        )
        original_result = original_benchmark.run_benchmark(pdf_files)
        
        # Run parallel pipeline
        parallel_benchmark = ParallelPipelineBenchmark(
            os.path.join(self.output_dir, "parallel"),
            workers=workers
        )
        parallel_result = await parallel_benchmark.run_benchmark(pdf_files)
        
        # Calculate comparison metrics
        speedup_factor = (original_result.total_time / parallel_result.total_time 
                         if parallel_result.total_time > 0 else 0)
        
        throughput_improvement = ((parallel_result.files_per_second - original_result.files_per_second)
                                / original_result.files_per_second * 100 
                                if original_result.files_per_second > 0 else 0)
        
        efficiency_gain = ((parallel_result.chunks_per_second - original_result.chunks_per_second)
                          / original_result.chunks_per_second * 100
                          if original_result.chunks_per_second > 0 else 0)
        
        memory_difference = parallel_result.memory_usage_mb - original_result.memory_usage_mb
        
        report = ComparisonReport(
            original_result=original_result,
            parallel_result=parallel_result,
            speedup_factor=speedup_factor,
            throughput_improvement=throughput_improvement,
            efficiency_gain=efficiency_gain,
            memory_difference_mb=memory_difference
        )
        
        # Save detailed results
        await self._save_benchmark_results(report)
        
        return report
    
    async def _save_benchmark_results(self, report: ComparisonReport):
        """Save benchmark results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        results_file = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Save human-readable summary
        summary_file = os.path.join(self.output_dir, f"benchmark_summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self._format_summary_report(report))
        
        logger.info(f"Benchmark results saved to: {results_file}")
        logger.info(f"Summary report saved to: {summary_file}")
    
    def _format_summary_report(self, report: ComparisonReport) -> str:
        """Format a human-readable summary report"""
        lines = []
        lines.append("=" * 80)
        lines.append("PDF EXTRACTION PIPELINE BENCHMARK RESULTS")
        lines.append("=" * 80)
        lines.append("")
        
        # Original pipeline results
        lines.append("ORIGINAL PIPELINE:")
        lines.append("-" * 40)
        orig = report.original_result
        lines.append(f"  Total files: {orig.total_files}")
        lines.append(f"  Processed: {orig.processed_files}")
        lines.append(f"  Failed: {orig.failed_files}")
        lines.append(f"  Total chunks: {orig.total_chunks}")
        lines.append(f"  Total time: {orig.total_time:.2f}s")
        lines.append(f"  Avg time per file: {orig.avg_time_per_file:.2f}s")
        lines.append(f"  Files per second: {orig.files_per_second:.2f}")
        lines.append(f"  Chunks per second: {orig.chunks_per_second:.2f}")
        lines.append(f"  Peak memory: {orig.memory_usage_mb:.1f}MB")
        lines.append(f"  Avg CPU: {orig.cpu_usage_percent:.1f}%")
        lines.append("")
        
        # Parallel pipeline results
        lines.append("PARALLEL PIPELINE:")
        lines.append("-" * 40)
        par = report.parallel_result
        lines.append(f"  Total files: {par.total_files}")
        lines.append(f"  Processed: {par.processed_files}")
        lines.append(f"  Failed: {par.failed_files}")
        lines.append(f"  Total chunks: {par.total_chunks}")
        lines.append(f"  Total time: {par.total_time:.2f}s")
        lines.append(f"  Avg time per file: {par.avg_time_per_file:.2f}s")
        lines.append(f"  Files per second: {par.files_per_second:.2f}")
        lines.append(f"  Chunks per second: {par.chunks_per_second:.2f}")
        lines.append(f"  Peak memory: {par.memory_usage_mb:.1f}MB")
        lines.append(f"  Avg CPU: {par.cpu_usage_percent:.1f}%")
        lines.append("")
        
        # Comparison metrics
        lines.append("PERFORMANCE COMPARISON:")
        lines.append("-" * 40)
        lines.append(f"  Speedup factor: {report.speedup_factor:.2f}x")
        lines.append(f"  Throughput improvement: {report.throughput_improvement:+.1f}%")
        lines.append(f"  Efficiency gain: {report.efficiency_gain:+.1f}%")
        lines.append(f"  Memory difference: {report.memory_difference_mb:+.1f}MB")
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 40)
        if report.speedup_factor > 2.0:
            lines.append("  ✓ Excellent speedup! Parallel pipeline is highly effective.")
        elif report.speedup_factor > 1.5:
            lines.append("  ✓ Good speedup. Parallel pipeline provides solid improvements.")
        elif report.speedup_factor > 1.0:
            lines.append("  ⚠ Modest speedup. Consider optimizing further or using original for small batches.")
        else:
            lines.append("  ✗ No speedup achieved. Investigate bottlenecks or use original pipeline.")
        
        if abs(report.memory_difference_mb) > 1000:
            lines.append(f"  ⚠ Significant memory difference ({report.memory_difference_mb:+.0f}MB)")
        
        # Error summary
        if orig.errors or par.errors:
            lines.append("")
            lines.append("ERRORS ENCOUNTERED:")
            lines.append("-" * 40)
            if orig.errors:
                lines.append(f"  Original pipeline: {len(orig.errors)} errors")
            if par.errors:
                lines.append(f"  Parallel pipeline: {len(par.errors)} errors")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)

def find_pdf_files(directory: str, max_files: int = None) -> List[str]:
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
    
    return pdf_files

def generate_test_data(output_dir: str, num_files: int = 5):
    """Generate synthetic test PDF files"""
    logger.info(f"Generating {num_files} test PDF files in {output_dir}")
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import lorem
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(num_files):
            filename = os.path.join(output_dir, f"test_document_{i+1:03d}.pdf")
            
            c = canvas.Canvas(filename, pagesize=letter)
            width, height = letter
            
            # Add multiple pages with varied content
            for page in range(3):
                c.drawString(100, height - 100, f"Test Document {i+1} - Page {page+1}")
                c.drawString(100, height - 140, f"Financial Report Sample Content")
                
                # Add some sample financial text
                y_pos = height - 200
                for line_num in range(20):
                    text = lorem.sentence() if hasattr(lorem, 'sentence') else f"Sample text line {line_num}"
                    c.drawString(100, y_pos, text[:80])
                    y_pos -= 20
                    if y_pos < 100:
                        break
                
                if page < 2:  # Don't add new page on last iteration
                    c.showPage()
            
            c.save()
            logger.info(f"Generated: {filename}")
        
        logger.info(f"Test data generation complete: {output_dir}")
        
    except ImportError:
        logger.error("reportlab not available for test data generation")
        logger.info("Install with: pip install reportlab")
        sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description='Benchmark Parallel PDF Extraction Pipeline')
    parser.add_argument('--pdf-dir', type=str, help='Directory containing PDF files to test')
    parser.add_argument('--max-files', type=int, default=10, help='Maximum number of files to test')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Output directory')
    parser.add_argument('--generate-test-data', action='store_true', help='Generate synthetic test data')
    parser.add_argument('--test-files', type=int, default=5, help='Number of test files to generate')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate test data if requested
    if args.generate_test_data:
        test_dir = os.path.join(args.output_dir, "test_data")
        generate_test_data(test_dir, args.test_files)
        if not args.pdf_dir:
            args.pdf_dir = test_dir
    
    if not args.pdf_dir:
        print("Error: Please specify --pdf-dir or use --generate-test-data")
        parser.print_help()
        sys.exit(1)
    
    # Find PDF files
    pdf_files = find_pdf_files(args.pdf_dir, args.max_files)
    
    if not pdf_files:
        logger.error(f"No PDF files found in: {args.pdf_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_files)} PDF files for benchmark")
    
    # Run benchmark
    runner = BenchmarkRunner(args.output_dir)
    report = await runner.run_comparison_benchmark(pdf_files, args.workers)
    
    # Print summary to console
    print(runner._format_summary_report(report))

if __name__ == "__main__":
    asyncio.run(main())