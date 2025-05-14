#!/usr/bin/env python3

import standalone
import os

cwd = os.getcwd()
compile_commands = os.path.join(os.environ["CODEQL_EXTRACTOR_CPP_LOG_DIR"], 'compile_commands.json')
extractor = os.path.join(os.environ['CODEQL_EXTRACTOR_CPP_ROOT'], 'tools', os.environ['CODEQL_PLATFORM'], 'extractor')
os.environ['SEMMLE_CPP_MISSING_INCLUDES_NOT_FATAL'] = '1'
os.environ['CODEQL_EXTRACTOR_CPP_OPTION_SCALE_TIMEOUTS'] = '10'

compiler = standalone.find_compiler()

if not standalone.is_enabled():
    print("Build mode 'none' is a pre-release feature that is not enabled in this configuration")
    exit(1)

if compiler is None:
    print('Error: no suitable C++ compiler was found. Check your PATH variable.')
    exit(1)

standalone.generate_compile_commands_json(cwd, compiler, compile_commands)
standalone.extract_compile_commands_json(extractor, compiler, compile_commands, standalone.get_thread_count())
