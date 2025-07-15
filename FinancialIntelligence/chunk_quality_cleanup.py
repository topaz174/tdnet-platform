#!/usr/bin/env python3
"""
Chunk Quality Cleanup

Post-processing script to clean up extracted chunks before embedding:
1. Strip leftover HTML/XML tags
2. Set vectorize=false for chunks with high token count or digit ratio
3. Analyze and report quality metrics

Usage:
    python chunk_quality_cleanup.py --analyze          # Analyze current chunks
    python chunk_quality_cleanup.py --cleanup          # Apply cleanup
    python chunk_quality_cleanup.py --preview          # Preview changes
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, Any, Tuple

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.unified_extraction_pipeline import DatabaseManager, load_database_config
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    sys.exit(1)

class ChunkQualityAnalyzer:
    """Analyzes and cleans up chunk quality issues"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
        # Quality thresholds
        self.max_tokens = 550
        self.max_digit_ratio = 0.40
        
        # HTML/XML tag pattern
        self.tag_pattern = re.compile(r'<[^>]*>')
        
    def analyze_chunks(self) -> Dict[str, Any]:
        """Analyze current chunk quality"""
        
        try:
            with self.db_manager.conn.cursor() as cur:
                # Overall statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(CASE WHEN vectorize = true THEN 1 END) as vectorizable,
                        COUNT(CASE WHEN tokens > %s THEN 1 END) as high_token,
                        COUNT(CASE WHEN is_numeric = true THEN 1 END) as numeric_chunks,
                        AVG(tokens) as avg_tokens,
                        AVG(char_length) as avg_length
                    FROM document_chunks
                """, (self.max_tokens,))
                
                stats = dict(zip([
                    'total_chunks', 'vectorizable', 'high_token', 'numeric_chunks', 'avg_tokens', 'avg_length'
                ], cur.fetchone()))
                
                # Check for HTML tags
                cur.execute("""
                    SELECT 
                        disclosure_id, chunk_index, content
                    FROM document_chunks 
                    WHERE content ~ '<[^>]*>'
                    LIMIT 100
                """)
                
                chunks_with_tags = cur.fetchall()
                
                # Calculate digit ratios for a sample
                cur.execute("""
                    SELECT content, tokens, char_length
                    FROM document_chunks 
                    WHERE vectorize = true
                    ORDER BY RANDOM()
                    LIMIT 1000
                """)
                
                sample_chunks = cur.fetchall()
                
                # Analyze digit ratios
                high_digit_ratio_count = 0
                digit_ratios = []
                
                for content, tokens, char_length in sample_chunks:
                    if content:
                        digit_ratio = self._calculate_digit_ratio(content)
                        digit_ratios.append(digit_ratio)
                        if digit_ratio > self.max_digit_ratio:
                            high_digit_ratio_count += 1
                
                return {
                    'statistics': stats,
                    'chunks_with_tags': len(chunks_with_tags),
                    'sample_tag_chunks': chunks_with_tags[:5],  # First 5 examples
                    'high_digit_ratio_sample': high_digit_ratio_count,
                    'avg_digit_ratio': sum(digit_ratios) / len(digit_ratios) if digit_ratios else 0,
                    'sample_size': len(sample_chunks)
                }
                
        except Exception as e:
            print(f"Error analyzing chunks: {e}")
            return {}
    
    def _calculate_digit_ratio(self, text: str) -> float:
        """Calculate ratio of digits to total characters"""
        if not text:
            return 0.0
        
        digit_count = sum(1 for char in text if char.isdigit())
        return digit_count / len(text)
    
    def preview_cleanup(self) -> Dict[str, int]:
        """Preview what would be cleaned up"""
        
        try:
            with self.db_manager.conn.cursor() as cur:
                # Count chunks that would be affected
                
                # 1. Chunks with HTML tags
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM document_chunks 
                    WHERE content ~ '<[^>]*>'
                """)
                chunks_with_tags = cur.fetchone()[0]
                
                # 2. High token chunks that are currently vectorizable
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM document_chunks 
                    WHERE tokens > %s AND vectorize = true
                """, (self.max_tokens,))
                high_token_vectorizable = cur.fetchone()[0]
                
                # 3. Get sample for digit ratio analysis
                cur.execute("""
                    SELECT id, content 
                    FROM document_chunks 
                    WHERE vectorize = true AND content IS NOT NULL
                """)
                
                all_vectorizable = cur.fetchall()
                high_digit_ratio_count = 0
                
                for chunk_id, content in all_vectorizable:
                    if self._calculate_digit_ratio(content) > self.max_digit_ratio:
                        high_digit_ratio_count += 1
                
                return {
                    'chunks_with_tags_to_clean': chunks_with_tags,
                    'high_token_to_disable': high_token_vectorizable,
                    'high_digit_ratio_to_disable': high_digit_ratio_count,
                    'total_vectorizable_chunks': len(all_vectorizable)
                }
                
        except Exception as e:
            print(f"Error previewing cleanup: {e}")
            return {}
    
    def cleanup_chunks(self, dry_run: bool = False) -> Dict[str, int]:
        """Clean up chunks based on quality criteria"""
        
        results = {
            'tags_cleaned': 0,
            'high_token_disabled': 0,
            'high_digit_ratio_disabled': 0,
            'total_affected': 0
        }
        
        try:
            with self.db_manager.conn.cursor() as cur:
                if not dry_run:
                    # 1. Strip HTML/XML tags
                    cur.execute("""
                        UPDATE document_chunks 
                        SET content = regexp_replace(content, '<[^>]*>', '', 'g'),
                            updated_at = NOW()
                        WHERE content ~ '<[^>]*>'
                    """)
                    results['tags_cleaned'] = cur.rowcount
                    
                    # 2. Disable vectorization for high token count chunks
                    cur.execute("""
                        UPDATE document_chunks 
                        SET vectorize = false,
                            updated_at = NOW()
                        WHERE tokens > %s AND vectorize = true
                    """, (self.max_tokens,))
                    results['high_token_disabled'] = cur.rowcount
                    
                    # 3. Disable vectorization for high digit ratio chunks
                    # This is more complex, so we'll do it in Python
                    cur.execute("""
                        SELECT id, content 
                        FROM document_chunks 
                        WHERE vectorize = true AND content IS NOT NULL
                    """)
                    
                    high_digit_chunks = []
                    for chunk_id, content in cur.fetchall():
                        if self._calculate_digit_ratio(content) > self.max_digit_ratio:
                            high_digit_chunks.append(chunk_id)
                    
                    if high_digit_chunks:
                        # Update in batches
                        for i in range(0, len(high_digit_chunks), 1000):
                            batch = high_digit_chunks[i:i+1000]
                            placeholders = ','.join(['%s'] * len(batch))
                            cur.execute(f"""
                                UPDATE document_chunks 
                                SET vectorize = false,
                                    updated_at = NOW()
                                WHERE id IN ({placeholders})
                            """, batch)
                            results['high_digit_ratio_disabled'] += cur.rowcount
                    
                    self.db_manager.conn.commit()
                
                else:
                    # Dry run - just count
                    cur.execute("SELECT COUNT(*) FROM document_chunks WHERE content ~ '<[^>]*>'")
                    results['tags_cleaned'] = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM document_chunks WHERE tokens > %s AND vectorize = true", (self.max_tokens,))
                    results['high_token_disabled'] = cur.fetchone()[0]
                    
                    # Count high digit ratio chunks
                    cur.execute("SELECT id, content FROM document_chunks WHERE vectorize = true AND content IS NOT NULL")
                    high_digit_count = 0
                    for chunk_id, content in cur.fetchall():
                        if self._calculate_digit_ratio(content) > self.max_digit_ratio:
                            high_digit_count += 1
                    results['high_digit_ratio_disabled'] = high_digit_count
                
                results['total_affected'] = (
                    results['tags_cleaned'] + 
                    results['high_token_disabled'] + 
                    results['high_digit_ratio_disabled']
                )
                
                return results
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
            if not dry_run:
                self.db_manager.conn.rollback()
            return results

def main():
    parser = argparse.ArgumentParser(description='Chunk Quality Cleanup Tool')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--analyze', action='store_true', help='Analyze current chunk quality')
    group.add_argument('--preview', action='store_true', help='Preview cleanup changes')
    group.add_argument('--cleanup', action='store_true', help='Apply cleanup (use --dry-run for preview)')
    
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    parser.add_argument('--max-tokens', type=int, default=550, help='Maximum tokens for vectorization')
    parser.add_argument('--max-digit-ratio', type=float, default=0.40, help='Maximum digit ratio for vectorization')
    
    args = parser.parse_args()
    
    # Load database configuration
    try:
        pg_dsn = load_database_config()
    except Exception as e:
        print(f"Error loading database configuration: {e}")
        sys.exit(1)
    
    # Connect to database
    db_manager = DatabaseManager(pg_dsn)
    if not db_manager.connect():
        print("Error: Failed to connect to database")
        sys.exit(1)
    
    try:
        analyzer = ChunkQualityAnalyzer(db_manager)
        analyzer.max_tokens = args.max_tokens
        analyzer.max_digit_ratio = args.max_digit_ratio
        
        if args.analyze:
            print("Analyzing chunk quality...")
            analysis = analyzer.analyze_chunks()
            
            if analysis:
                print("\n" + "="*60)
                print("CHUNK QUALITY ANALYSIS")
                print("="*60)
                
                stats = analysis['statistics']
                print(f"Total chunks: {stats['total_chunks']:,}")
                print(f"Vectorizable chunks: {stats['vectorizable']:,}")
                print(f"High token chunks (>{args.max_tokens}): {stats['high_token']:,}")
                print(f"Numeric chunks: {stats['numeric_chunks']:,}")
                print(f"Average tokens: {stats['avg_tokens']:.1f}")
                print(f"Average length: {stats['avg_length']:.1f}")
                
                print(f"\nQuality Issues:")
                print(f"Chunks with HTML tags: {analysis['chunks_with_tags']:,}")
                print(f"High digit ratio (sample): {analysis['high_digit_ratio_sample']:,}/{analysis['sample_size']:,}")
                print(f"Average digit ratio: {analysis['avg_digit_ratio']:.3f}")
                
                if analysis['sample_tag_chunks']:
                    print(f"\nSample chunks with tags:")
                    for disclosure_id, chunk_index, content in analysis['sample_tag_chunks']:
                        tags = re.findall(r'<[^>]*>', content)
                        print(f"  Disclosure {disclosure_id}, chunk {chunk_index}: {tags[:3]}...")
        
        elif args.preview:
            print("Previewing cleanup changes...")
            preview = analyzer.preview_cleanup()
            
            if preview:
                print("\n" + "="*60)
                print("CLEANUP PREVIEW")
                print("="*60)
                print(f"Chunks with tags to clean: {preview['chunks_with_tags_to_clean']:,}")
                print(f"High token chunks to disable: {preview['high_token_to_disable']:,}")
                print(f"High digit ratio chunks to disable: {preview['high_digit_ratio_to_disable']:,}")
                print(f"Total vectorizable chunks: {preview['total_vectorizable_chunks']:,}")
                
                total_affected = (
                    preview['chunks_with_tags_to_clean'] +
                    preview['high_token_to_disable'] +
                    preview['high_digit_ratio_to_disable']
                )
                print(f"\nTotal chunks to be modified: {total_affected:,}")
        
        elif args.cleanup:
            action = "Previewing" if args.dry_run else "Applying"
            print(f"{action} cleanup...")
            
            results = analyzer.cleanup_chunks(dry_run=args.dry_run)
            
            print("\n" + "="*60)
            print(f"CLEANUP {'PREVIEW' if args.dry_run else 'RESULTS'}")
            print("="*60)
            print(f"Tags cleaned: {results['tags_cleaned']:,}")
            print(f"High token chunks disabled: {results['high_token_disabled']:,}")
            print(f"High digit ratio chunks disabled: {results['high_digit_ratio_disabled']:,}")
            print(f"Total affected: {results['total_affected']:,}")
            
            if args.dry_run:
                print(f"\nRun without --dry-run to apply these changes.")
            else:
                print(f"\n✓ Cleanup completed successfully!")
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    main()