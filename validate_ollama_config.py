#!/usr/bin/env python3
"""
Simple validation script to check LLM configuration logic without requiring full Django setup.
This validates that the Ollama model configuration is correct.
"""

def validate_ollama_config():
    """Validate that Ollama models are properly configured in apps.py"""
    
    print("=" * 60)
    print("Validating Ollama LLM Configuration")
    print("=" * 60)
    
    # Read apps.py and verify Ollama configuration
    apps_py_path = '/home/runner/work/AquiLLM/AquiLLM/aquillm/aquillm/apps.py'
    
    with open(apps_py_path, 'r') as f:
        content = f.read()
    
    # Check for Ollama model configurations
    ollama_checks = {
        'GEMMA3': {
            'model': 'ebdm/gemma3-enhanced:12b',
            'base_url': 'http://ollama:11434/v1/',
            'interface': 'OpenAIInterface'
        },
        'LLAMA3.2': {
            'model': 'llama3.2',
            'base_url': 'http://ollama:11434/v1/',
            'interface': 'OpenAIInterface'
        },
        'GPT-OSS': {
            'model': 'gpt-oss:120b',
            'base_url': 'http://ollama:11434/v1/',
            'interface': 'OpenAIInterface'
        }
    }
    
    all_passed = True
    
    for model_name, expected in ollama_checks.items():
        print(f"\n✓ Checking {model_name} configuration...")
        
        # Check if model is in the file
        if f"llm_choice == '{model_name}'" in content:
            print(f"  ✓ {model_name} choice found")
        else:
            print(f"  ✗ {model_name} choice NOT found")
            all_passed = False
            continue
        
        # Check if correct model string is used
        if expected['model'] in content:
            print(f"  ✓ Model string '{expected['model']}' found")
        else:
            print(f"  ✗ Model string '{expected['model']}' NOT found")
            all_passed = False
        
        # Check if Ollama base URL is used
        if expected['base_url'] in content:
            print(f"  ✓ Ollama base URL '{expected['base_url']}' found")
        else:
            print(f"  ✗ Ollama base URL '{expected['base_url']}' NOT found")
            all_passed = False
        
        # Check if OpenAIInterface is used
        if expected['interface'] in content:
            print(f"  ✓ Using {expected['interface']}")
        else:
            print(f"  ✗ {expected['interface']} NOT found")
            all_passed = False
    
    # Additional check: Verify no Bedrock configuration
    print(f"\n✓ Checking AWS Bedrock removal...")
    if 'BEDROCK-CLAUDE' in content or 'async_anthropic_bedrock_client' in content:
        print(f"  ✗ AWS Bedrock configuration still present (should be removed)")
        all_passed = False
    else:
        print(f"  ✓ AWS Bedrock configuration successfully removed")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("Ollama local LLM models are correctly configured.")
    else:
        print("✗ SOME CHECKS FAILED!")
        print("Please review the configuration.")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    import sys
    passed = validate_ollama_config()
    sys.exit(0 if passed else 1)
