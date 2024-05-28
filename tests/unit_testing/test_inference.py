""" 

This script is used to test the inference of the SciNoBo-RAA bulk inference.

"""
import unittest
# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #
from citance_analysis.pipeline.inference import main

import argparse
from unittest.mock import patch

class TestInference(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main(self, mock_print, mock_args):
        # Set up test data
        mock_args.return_value = argparse.Namespace(
            input_dir='examples/input/', 
            output_dir='examples/output/',
            xml_mode=False
        )
        
        # Call the main function
        main()
        
        # Assert that the expected output is printed
        mock_print.assert_called_with("Done!")

if __name__ == '__main__':
    unittest.main()