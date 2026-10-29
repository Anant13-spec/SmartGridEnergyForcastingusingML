"""
Automated integration test for Streamlit Dashboard using AppTest.
"""

import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class TestStreamlitApp(unittest.TestCase):

    def test_app_renders_without_exceptions(self):
        """Verify that the dashboard loads completely without unhandled exceptions."""
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=25)
        at.run()
        
        self.assertEqual(
            len(at.exception), 0,
            f"App threw unhandled exceptions: {at.exception}"
        )


if __name__ == "__main__":
    unittest.main()
