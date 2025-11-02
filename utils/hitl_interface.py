"""
Human-in-the-Loop (HITL) Interface for the MAS framework.
Provides transport-agnostic interface for user interactions with audit logging.
"""

import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class HitlInterface(ABC):
    """Abstract base class for Human-in-the-Loop interactions."""
    
    def __init__(self, audit_log_path: Optional[str] = None):
        """
        Initialize HITL interface with optional audit logging.
        Args:
            audit_log_path: Path to audit log file. If None, uses default location.
        """
        self.audit_log_path = audit_log_path or "logs/hitl_audit.json"
        self._ensure_audit_dir()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _ensure_audit_dir(self):
        """Ensure audit log directory exists."""
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
    
    def _log_audit(self, event_type: str, context: Dict[str, Any], 
                   user_input: Optional[str] = None, system_response: Optional[str] = None):
        """Log HITL interaction to audit file."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            "context": context,
            "user_input": user_input,
            "system_response": system_response
        }
        
        # Append to audit log file
        try:
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            logging.warning(f"Failed to write audit log: {e}")
    
    @abstractmethod
    def prompt_user(self, message: str, options: Optional[List[str]] = None, 
                   multi_select: bool = False) -> Union[str, List[str]]:
        """
        Prompt user for input.
        Args:
            message: The prompt message
            options: List of available options (for selection)
            multi_select: Whether multiple selections are allowed
        Returns:
            User input (string or list of strings)
        """
        pass
    
    @abstractmethod
    def show_info(self, message: str, data: Optional[Any] = None):
        """
        Show information to user.
        Args:
            message: Information message
            data: Optional data to display (DataFrame, dict, etc.)
        """
        pass
    
    def prompt_with_audit(self, message: str, options: Optional[List[str]] = None, 
                         multi_select: bool = False, context: Optional[Dict[str, Any]] = None) -> Union[str, List[str]]:
        """
        Prompt user with audit logging.
        Args:
            message: The prompt message
            options: List of available options
            multi_select: Whether multiple selections are allowed
            context: Additional context for audit logging
        Returns:
            User input
        """
        context = context or {}
        context.update({
            "prompt_message": message,
            "options": options,
            "multi_select": multi_select
        })
        
        # Log the prompt
        self._log_audit("prompt", context)
        
        # Get user input
        user_input = self.prompt_user(message, options, multi_select)
        
        # Log the response
        self._log_audit("response", context, user_input=str(user_input))
        
        return user_input
    
    def show_info_with_audit(self, message: str, data: Optional[Any] = None, 
                           context: Optional[Dict[str, Any]] = None):
        """
        Show information with audit logging.
        Args:
            message: Information message
            data: Optional data to display
            context: Additional context for audit logging
        """
        context = context or {}
        context.update({
            "info_message": message,
            "data_type": type(data).__name__ if data is not None else None
        })
        
        # Log the info display
        self._log_audit("info_display", context, system_response=message)
        
        # Show the information
        self.show_info(message, data)


class CliHitlInterface(HitlInterface):
    """CLI-based HITL interface implementation."""
    
    def prompt_user(self, message: str, options: Optional[List[str]] = None, 
                   multi_select: bool = False) -> Union[str, List[str]]:
        """Prompt user via command line interface."""
        print(f"\n{message}")
        
        if options:
            print("Available options:")
            for i, option in enumerate(options):
                print(f"  [{i}] {option}")
            
            if multi_select:
                while True:
                    try:
                        indices_input = input("Enter comma-separated numbers for your choices: ").strip()
                        if not indices_input:
                            return []
                        indices = [int(i.strip()) for i in indices_input.split(',')]
                        if all(0 <= i < len(options) for i in indices):
                            return [options[i] for i in indices]
                        else:
                            print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter comma-separated numbers.")
            else:
                while True:
                    try:
                        index = int(input("Enter the number of your choice: ").strip())
                        if 0 <= index < len(options):
                            return options[index]
                        else:
                            print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        else:
            return input("Your input: ").strip()
    
    def show_info(self, message: str, data: Optional[Any] = None):
        """Show information via command line interface."""
        print(f"\n{message}")
        if data is not None:
            if hasattr(data, 'to_string'):
                print(data.to_string())
            elif isinstance(data, (dict, list)):
                print(json.dumps(data, indent=2, default=str))
            else:
                print(str(data))


class WebHitlInterface(HitlInterface):
    """Web-based HITL interface implementation (placeholder for future web UI)."""
    
    def __init__(self, audit_log_path: Optional[str] = None, **kwargs):
        super().__init__(audit_log_path)
        self.pending_prompts = []
        self.responses = {}
        logging.info("Web HITL interface initialized (placeholder implementation)")
    
    def prompt_user(self, message: str, options: Optional[List[str]] = None, 
                   multi_select: bool = False) -> Union[str, List[str]]:
        """Placeholder for web-based prompting."""
        # In a real implementation, this would send the prompt to a web UI
        # and wait for the response via websocket or polling
        prompt_id = f"prompt_{len(self.pending_prompts)}"
        self.pending_prompts.append({
            "id": prompt_id,
            "message": message,
            "options": options,
            "multi_select": multi_select
        })
        
        logging.warning(f"Web HITL not implemented. Using fallback CLI for: {message}")
        # Fallback to CLI for now
        cli = CliHitlInterface()
        return cli.prompt_user(message, options, multi_select)
    
    def show_info(self, message: str, data: Optional[Any] = None):
        """Placeholder for web-based info display."""
        logging.info(f"Web info display (placeholder): {message}")
        if data is not None:
            logging.info(f"Data: {type(data)} - {str(data)[:200]}...")


def get_hitl_interface(interface_type: str = "cli", **kwargs) -> HitlInterface:
    """
    Factory function to get HITL interface instance.
    Args:
        interface_type: Type of interface ("cli" or "web")
        **kwargs: Additional arguments for interface initialization
    Returns:
        HitlInterface instance
    """
    if interface_type.lower() == "cli":
        return CliHitlInterface(**kwargs)
    elif interface_type.lower() == "web":
        return WebHitlInterface(**kwargs)
    else:
        raise ValueError(f"Unknown HITL interface type: {interface_type}")


if __name__ == "__main__":
    # Test the HITL interface
    hitl = get_hitl_interface("cli")
    
    # Test single selection
    choice = hitl.prompt_with_audit(
        "Choose a problem type:",
        options=["classification", "regression", "anomaly_detection"],
        context={"test": "single_selection"}
    )
    print(f"User chose: {choice}")
    
    # Test multi-selection
    choices = hitl.prompt_with_audit(
        "Select features to use:",
        options=["feature1", "feature2", "feature3", "feature4"],
        multi_select=True,
        context={"test": "multi_selection"}
    )
    print(f"User chose: {choices}")
    
    # Test info display
    hitl.show_info_with_audit("Test information", {"key": "value"}, {"test": "info_display"})
