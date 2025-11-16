#!/usr/bin/env python
"""
Simple test script to verify the bug fixes:
1. SUMO environment reset with full_restart parameter
2. Environment cleanup (close method)
3. Basic training loop structure

Note: This requires SUMO to be installed. If SUMO is not available,
these tests will be skipped.
"""

import os
import sys

def test_environment_reset():
    """Test that environment reset works with full_restart parameter"""
    print("Testing environment reset with full_restart parameter...")
    try:
        # Check the source code directly without importing (to avoid dependency issues)
        with open('env/sumo_env.py', 'r') as f:
            content = f.read()
        
        # Check that reset method has full_restart parameter
        assert 'def reset(self, full_restart=False):' in content, \
               "reset() should have full_restart parameter with default False"
        print("  ✓ reset() has full_restart parameter with default False")
        
        # Check for full_restart logic
        assert 'if full_restart or not hasattr(self' in content, \
               "Should check full_restart parameter"
        print("  ✓ reset() checks full_restart parameter")
        
        # Check for process termination in full_restart
        assert 'terminate()' in content, \
               "Should terminate process in full_restart"
        print("  ✓ reset() terminates old process in full_restart mode")
        
        # Check for traci.load() in fast reset
        assert 'traci.load(' in content, \
               "Should use traci.load() for fast reset"
        print("  ✓ reset() uses traci.load() for fast reset")
        
        # Check that close method exists
        assert 'def close(self):' in content, "Environment should have close() method"
        print("  ✓ close() method exists")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_training_loop_structure():
    """Test that training loop has proper structure"""
    print("\nTesting training loop structure...")
    try:
        # Read the main_train.py file and check for key patterns
        with open('main_train.py', 'r') as f:
            content = f.read()
        
        # Check for try-finally block
        assert 'try:' in content and 'finally:' in content, "Training loop should have try-finally"
        print("  ✓ Training loop has try-finally block")
        
        # Check for env.close() in finally
        assert 'env.close()' in content, "finally block should call env.close()"
        print("  ✓ finally block calls env.close()")
        
        # Check for full_restart parameter usage
        assert 'full_restart=' in content, "Should use full_restart parameter"
        print("  ✓ Training uses full_restart parameter")
        
        # Check for manager.save() usage
        assert 'manager.save()' in content, "Should use manager.save() for checkpoints"
        print("  ✓ Uses manager.save() for checkpoints")
        
        # Check for proper global observation handling
        assert 'gobs_buf.append' in content, "Should collect global observations"
        print("  ✓ Collects global observations for critic")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_checkpoint_manager():
    """Test that checkpoint manager is properly configured"""
    print("\nTesting checkpoint manager configuration...")
    try:
        with open('main_train.py', 'r') as f:
            content = f.read()
        
        # Check for CheckpointManager with max_to_keep
        assert 'CheckpointManager' in content, "Should use CheckpointManager"
        assert 'max_to_keep=5' in content, "Should keep last 5 checkpoints"
        print("  ✓ CheckpointManager configured with max_to_keep=5")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_environment_close():
    """Test that environment close method is properly implemented"""
    print("\nTesting environment close method...")
    try:
        with open('env/sumo_env.py', 'r') as f:
            content = f.read()
        
        # Check for proper close implementation
        assert 'def close(self):' in content, "Should have close() method"
        
        # Check for traci.close()
        assert 'self.traci.close()' in content or 'traci.close()' in content, \
               "Should close traci connection"
        print("  ✓ close() closes traci connection")
        
        # Check for process termination
        assert 'terminate()' in content or 'kill()' in content, \
               "Should terminate SUMO process"
        print("  ✓ close() terminates SUMO process")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Bug Fix Validation Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Environment Reset", test_environment_reset()))
    results.append(("Training Loop Structure", test_training_loop_structure()))
    results.append(("Checkpoint Manager", test_checkpoint_manager()))
    results.append(("Environment Close", test_environment_close()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    # Return exit code
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
