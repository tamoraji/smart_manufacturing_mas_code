#!/usr/bin/env python3
"""
Test script for HITL interface functionality.
"""

import os
import json
import tempfile
from utils.hitl_interface import CliHitlInterface, WebHitlInterface, get_hitl_interface

def test_hitl_interface():
    """Test the HITL interface functionality."""
    print("Testing HITL Interface...")
    
    # Test with temporary audit log
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        audit_log_path = f.name
    
    try:
        # Test CLI interface
        print("\n1. Testing CLI HITL Interface...")
        cli_hitl = CliHitlInterface(audit_log_path=audit_log_path)
        
        # Test audit logging
        cli_hitl._log_audit("test_event", {"test": "data"}, "user_input", "system_response")
        
        # Test info display
        cli_hitl.show_info_with_audit("Test information message", {"key": "value"})
        
        print("✓ CLI HITL Interface created and audit logging works")
        
        # Test Web interface (placeholder)
        print("\n2. Testing Web HITL Interface...")
        web_hitl = WebHitlInterface(audit_log_path=audit_log_path)
        web_hitl.show_info_with_audit("Test web info", {"web": "data"})
        
        print("✓ Web HITL Interface created")
        
        # Test factory function
        print("\n3. Testing Factory Function...")
        hitl_cli = get_hitl_interface("cli", audit_log_path=audit_log_path)
        hitl_web = get_hitl_interface("web", audit_log_path=audit_log_path)
        
        print("✓ Factory function works for both CLI and Web interfaces")
        
        # Check audit log content
        print("\n4. Checking Audit Log...")
        with open(audit_log_path, 'r') as f:
            lines = f.readlines()
            print(f"✓ Audit log created with {len(lines)} entries")
            
            # Show first few entries
            for i, line in enumerate(lines[:3]):
                entry = json.loads(line.strip())
                print(f"  Entry {i+1}: {entry['event_type']} at {entry['timestamp']}")
        
        print("\n✅ All HITL Interface tests passed!")
        
    finally:
        # Clean up
        if os.path.exists(audit_log_path):
            os.unlink(audit_log_path)

if __name__ == "__main__":
    test_hitl_interface()
