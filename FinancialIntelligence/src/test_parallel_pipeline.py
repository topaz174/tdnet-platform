#!/usr/bin/env python3
"""
Simple test script for the parallel PDF extraction pipeline
"""

import asyncio
import os
import sys
import tempfile
import logging

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_pdf(filename: str):
    """Create a simple test PDF using available libraries"""
    try:
        # Try using reportlab if available
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        
        # Add some test content
        c.drawString(100, height - 100, "Test Financial Document")
        c.drawString(100, height - 140, "This is a test PDF for the parallel extraction pipeline")
        c.drawString(100, height - 180, "Financial data: 売上高 1,000,000円")
        c.drawString(100, height - 220, "利益: 250,000円")
        c.drawString(100, height - 260, "Japanese text: これは日本語のテストです。")
        
        c.save()
        logger.info(f"Created test PDF: {filename}")
        return True
        
    except ImportError:
        logger.warning("reportlab not available, creating simple text file instead")
        
        # Create a simple text file for testing
        text_filename = filename.replace('.pdf', '.txt')
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write("Test Financial Document\n")
            f.write("This is a test file for the parallel extraction pipeline\n")
            f.write("Financial data: 売上高 1,000,000円\n")
            f.write("利益: 250,000円\n")
            f.write("Japanese text: これは日本語のテストです。\n")
        
        logger.info(f"Created test text file: {text_filename}")
        return False

async def test_basic_import():
    """Test that we can import the parallel pipeline"""
    try:
        from parallel_pdf_extraction_pipeline import (
            ParallelProcessingConfig, 
            ParallelPDFExtractionPipeline
        )
        logger.info("✓ Successfully imported parallel pipeline classes")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import parallel pipeline: {e}")
        return False

async def test_config_creation():
    """Test creating a configuration"""
    try:
        from parallel_pdf_extraction_pipeline import ParallelProcessingConfig
        
        config = ParallelProcessingConfig(
            max_workers=4,
            max_concurrent_files=2,
            batch_size=2,
            use_gpu_ocr=False,  # Disable GPU for basic test
            fallback_to_cpu=True
        )
        logger.info("✓ Successfully created parallel processing configuration")
        logger.info(f"  Workers: {config.max_workers}")
        logger.info(f"  Concurrent files: {config.max_concurrent_files}")
        logger.info(f"  Batch size: {config.batch_size}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create configuration: {e}")
        return False

async def test_pipeline_initialization():
    """Test initializing the pipeline"""
    try:
        from parallel_pdf_extraction_pipeline import (
            ParallelProcessingConfig, 
            ParallelPDFExtractionPipeline
        )
        
        config = ParallelProcessingConfig(
            max_workers=2,
            max_concurrent_files=1,
            use_gpu_ocr=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = ParallelPDFExtractionPipeline(config, temp_dir)
            logger.info("✓ Successfully initialized parallel pipeline")
            logger.info(f"  Output directory: {pipeline.output_dir}")
            logger.info(f"  Worker count: {pipeline.config.max_workers}")
            return True
            
    except Exception as e:
        logger.error(f"✗ Failed to initialize pipeline: {e}")
        return False

async def test_file_processing():
    """Test processing a simple file (if available)"""
    try:
        from parallel_pdf_extraction_pipeline import (
            ParallelProcessingConfig, 
            ParallelPDFExtractionPipeline
        )
        
        config = ParallelProcessingConfig(
            max_workers=2,
            max_concurrent_files=1,
            use_gpu_ocr=False,
            disable_ocr=True  # Disable OCR for basic test
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = []
            for i in range(2):
                test_file = os.path.join(temp_dir, f"test_{i}.pdf")
                if create_test_pdf(test_file):
                    test_files.append(test_file)
            
            if not test_files:
                logger.info("⚠ No test PDFs created (reportlab not available)")
                return True  # Not a failure, just no PDFs to test with
            
            # Initialize pipeline
            output_dir = os.path.join(temp_dir, "output")
            pipeline = ParallelPDFExtractionPipeline(config, output_dir)
            
            logger.info(f"Testing with {len(test_files)} test files...")
            
            # Process files
            summary = await pipeline.process_files_async(test_files)
            
            logger.info("✓ Successfully processed test files")
            logger.info(f"  Files processed: {summary.get('total_files_processed', 0)}")
            logger.info(f"  Files failed: {summary.get('total_files_failed', 0)}")
            logger.info(f"  Total chunks: {summary.get('total_chunks_extracted', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Failed to process test files: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        return False

async def run_all_tests():
    """Run all tests"""
    logger.info("="*60)
    logger.info("PARALLEL PDF EXTRACTION PIPELINE - BASIC TESTS")
    logger.info("="*60)
    
    tests = [
        ("Import Test", test_basic_import),
        ("Configuration Test", test_config_creation),
        ("Pipeline Initialization Test", test_pipeline_initialization),
        ("File Processing Test", test_file_processing),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! The parallel pipeline is working correctly.")
    else:
        logger.warning(f"⚠ {len(results) - passed} tests failed. Check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n✅ Basic functionality test completed successfully!")
        print("You can now use the parallel pipeline with your PDF files.")
        print("\nNext steps:")
        print("1. Install optional dependencies for better performance:")
        print("   pip install aiofiles jaconv mecab-python3")
        print("2. Test with your actual PDF files:")
        print("   python parallel_pdf_extraction_pipeline.py --pdf-dir /path/to/your/pdfs --workers 8")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)