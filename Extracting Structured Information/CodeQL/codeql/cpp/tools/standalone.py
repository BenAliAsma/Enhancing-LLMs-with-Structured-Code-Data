# Library for standalone extraction
# Not intended to be run directly

import os
import subprocess
import json
import tempfile
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import pathlib
import shutil
import shlex

import dir_trie

verbose_logging = False      # Whether to output launched processes
prefer_cl = True             # If cl and clang are both available, which do we use

def shlex_join(split_command):
    '''
    Implements shlex.join() for platforms that don't support it
    '''
    return ' '.join(shlex.quote(x) for x in split_command)

def is_absolute(path):
    '''
    check if the given path is an absolute path in either posix or windows.
    '''
    windows_path = pathlib.PureWindowsPath(path)
    posix_path = pathlib.PurePosixPath(path)
    return windows_path.is_absolute() or posix_path.is_absolute()

class CompileCommands:
    '''
    Class to construct the compile_commands.json file
    '''
    def __init__(self, repo_dir, compiler):
        self.compiler = compiler
        self.cpp_compiler = compiler.get_cpp_compiler_prefix()
        self.c_compiler = compiler.get_c_compiler_prefix()
        self.trie = dir_trie.DirTrie()

        self.repo_dir = repo_dir
        self.source_files = set()
        self.cpp_endings = ['.c', '.cpp', '.cc', '.cxx', '.c++']
        self.all_files = set()
        self.include_files = {}


    def find_files(self):
        '''Index all files in self.repo_dir'''
        self.trie.of_dir(self.repo_dir)

        for path in self.trie.get_paths():
            fullname = os.path.join(*path)
            self.all_files.add(fullname)
            _, file_extension = os.path.splitext(fullname)
            if file_extension.lower() in self.cpp_endings:
                self.source_files.add(fullname)

        print("Found", len(self.source_files), "source files", len(self.all_files), "total files")


    def find_include_paths(self, include_file):
        '''
        Find all possible solutions that can satisfy an include statement.
        Relative includes (includes starting with '.') will return none.
        '''
        return self.trie.lookup(include_file)

    # Regular expression for an include directive
    # Capture everything before the first / or \
    include_re = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]')

    def get_include_files(self, src):
        '''Get all files referenced in include preprocessor statements'''
        includes = self.include_files.get(src)
        if includes is not None:
            return includes

        includes=[]
        with open(src, 'r', errors='ignore', encoding='utf-8') as f:
            for line in f:
                m = self.include_re.match(line)
                if m:
                    include = m.group(1)
                    # Map '\' to '/' to be more platform agnostic.
                    include = include.replace('\\', '/')
                    includes.append(include)
                else:
                    #print("Error: Include not matched!")
                    pass

        self.include_files[src] = includes
        return includes

    def file_exists(self, fullpath):
        '''Test if a file exists'''
        return fullpath in self.all_files


    def include_dir_distance(self, src, include_dir):
        '''
        Calculate a distances score for the include dir relative to the src
        The algorithm will prioritize any include directory in a subdirectory of the source file
        '''
        common_path = os.path.commonpath([src, include_dir])
        common_dirs = common_path.count(os.path.sep)
        # Include ending directory in the count if any
        if common_path[-1] != os.path.sep:
            common_dirs += 1

        src_dirs = src.count(os.path.sep) - 1 # Will always have a file also
        include_dirs = include_dir.count(os.path.sep)

        return (src_dirs - common_dirs) * 1000 + (include_dirs - common_dirs)

    def resolve_include(self, src, include_file):
        '''
        Resolve an include based on file its included from.
        Return the file to be included and the directory that should be used to include the file,
        or none if the is no solution
        '''

        dirs = self.find_include_paths(include_file)
        if dirs == []:
            return None

        include_dirs = []
        for d in dirs:
            fulldir = os.path.join(*d) # This is the dir to be included.
            fullpath = os.path.join(fulldir, include_file)
            distance = self.include_dir_distance(src, fulldir)
            include_dirs.append((distance, fulldir, fullpath))

        return min(include_dirs)[1:]

    def lookup_include(self, include_file, include_dirs):
        '''
        Find the full path of the include file by searching though the given include paths
        The function returns the absolute path to the include file
        or None if the include path cannot be found
        '''
        for d in include_dirs:
            fullpath = os.path.abspath(os.path.join(d, include_file))
            if self.file_exists(fullpath):
                return fullpath

        return None

    def get_include_dirs(self, src, seen=None, include_dirs=None):
        '''
        Gets the set of include directories needed to compile a source file.
        The function processes includes transitivly, and mimics the compilers
        behaviour when resolving includes. The function returns an ordered list
        of include paths for the given source file.

        '''

        if include_dirs is None:
            include_dirs = []

        if seen is None:
            seen = set()

        includes = self.get_include_files(src)

        # We do not process include files specified through an absolute path
        for include in [path for path in includes if not is_absolute(path)]:
            include_file = self.lookup_include(include, include_dirs)
            # Could we find it?
            if include_file is None:

                # Dont try to resolve relative files by traversing include directories.
                if include[0] == '.':
                    continue

                inc = self.resolve_include(src, include)
                if inc is None:
                    continue

                (include_dir, include_file) = inc
                include_dirs.append(include_dir)

            if include_file not in seen:
                seen.add(include_file)
                include_dirs = self.get_include_dirs(include_file,
                                                     seen=seen,
                                                     include_dirs=include_dirs)

        return include_dirs


    def get_compile_commands(self):
        '''
        Returns the list of all compile commands.
        '''
        compile_commands = []
        for src in sorted(self.source_files):
            cmd_line = []
            if src.lower().endswith('.c'):
                cmd_line += self.c_compiler
            else:
                cmd_line += self.cpp_compiler

            # Add include directives needed to satisfy include statements.
            for i in self.get_include_dirs(src):
                cmd_line.append(self.compiler.include(i))

            out = src + ".o"
            compile_command = {}
            compile_command['directory'] = self.repo_dir
            compile_command['command'] = shlex_join(cmd_line + self.compiler.output(out) + self.compiler.source(src))
            compile_command['output'] = out
            compile_command['file'] = src
            compile_commands.append(compile_command)
        return compile_commands

    def write_compile_commands(self, compile_commands, output):
        '''
        Write compile_commands.json to the given file.
        '''
        encoder = json.JSONEncoder(indent=4)

        with open(output, "w") as text_file:
            text_file.write(encoder.encode(compile_commands))

        print('Written', output)

def generate_compile_commands_json(dir, compiler, json_file):
    '''
    Compute and write the compile_commands to the given json file.
    '''
    print('Generating compilation commands...')
    cc = CompileCommands(dir, compiler)
    cc.find_files()
    commands = cc.get_compile_commands()
    cc.write_compile_commands(commands, json_file)

class CommandWithResponse:
    '''
    A command line split into the command and its arguments.
    The command must support response files - the ability to replace its arguments
    with an `@` file containing the arguments.
    '''
    def __init__(self, line, compiler):
        l = shlex.split(line)
        self.command = l[:1] if len(l)>0 else []
        self.args = l[1:] if len(l)>1 else []
        self.compiler = compiler

    def write_response_file(self, response_file):
        self.compiler.write_response_file(response_file, self.args)

class ProcessWithResponseFile:
    '''
    Helper to rewrite a command line to a temporary response file, then launch the process.
    '''
    def __init__(self, command: CommandWithResponse):
        cmd = command.command
        self.response_file = tempfile.NamedTemporaryFile(delete=False)
        command.write_response_file(self.response_file)
        self.response_file.close()
        if verbose_logging:
            print('Running:', cmd, '@'+self.response_file.name)
        # Don't capture stdout or stderr as these will be written to the log directory of the extractor.
        self.process = subprocess.Popen(cmd +  ['@'+self.response_file.name], shell=False,
            stdout=None if verbose_logging else subprocess.DEVNULL,
            stderr=None if verbose_logging else subprocess.DEVNULL)

    def wait(self):
        self.process.wait()
        self.returncode = self.process.returncode
        # print('Return code:', self.returncode)
        os.unlink(self.response_file.name)

def run_process_with_response_file(command):
    try:
        p = ProcessWithResponseFile(command)
        p.wait()
        return p.returncode==0
    except Exception as ex:
        print('Failed to launch', command.command[0], 'because', ex)
        return False

def run_commands_in_parallel(commands, threads):
    print('Running', len(commands), 'commands in', threads, 'threads')
    errors = 0
    successes = 0
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(run_process_with_response_file, command): command for command in commands}
        for future in as_completed(futures):
            output = futures[future]
            try:
                if future.result():
                    successes = successes + 1
                else:
                    errors = errors + 1
            except Exception as e:
                print(f"{output}: {e}")
                errors = errors + 1
    print('Ran', len(commands), 'commands,', successes, 'succeeded,', errors, 'failed')

def run_compile_commands_json(compiler, json_file, threads):
    '''
    Read a list of compile-commands from the given json file, then execute them in parallel.
    This should be run in a traced context.
    '''
    print('Running compile commands')
    with open(json_file, 'r') as f:
        data = f.read()
    compile_commands = json.loads(data)
    compiler_invocations = []
    for command in compile_commands:
        compiler_invocations.append(CommandWithResponse(command['command'], compiler))
    run_commands_in_parallel(list, threads)

def extract_compile_commands_json(extractor, compiler, json_file, threads):
    '''
    Read the list of compile commands and then launch the extractor directly
    for each command. This should not be run in a traced context.
    '''
    print('Extracting compile commands')
    with open(json_file, 'r') as f:
        data = f.read()
    commands = json.loads(data)
    list = []
    for command in commands:
        command = CommandWithResponse(command['command'], compiler)
        command.command = [extractor, '--mimic'] + command.command
        list.append(command)
    run_commands_in_parallel(list, threads)

class ClangCompiler:
    '''
    Constructs the command line for the clang compiler.
    '''
    def __init__(self, clang, clangpp):
        self.clang = clang
        self.clangpp = clangpp

    def get_cpp_compiler_prefix(self):
        return [self.clangpp, '-std=c++20', '-fexceptions', '-c', '-fsyntax-only']

    def get_c_compiler_prefix(self):
        return [self.clang, '-c', '-fsyntax-only']

    def include(self, inc):
        return '-I' + str(inc)

    def output(self, out):
        return ['-o',out]

    def source(self, src):
        return [src]

    def write_response_file(self, response_file, args):
        response_file.write(shlex_join(args).encode('utf-8'))

class MsvcCompiler:
    '''
    Constructs the command line for the MSVC compiler.
    '''
    def __init__(self, cl):
        self.cl = cl

    def get_cpp_compiler_prefix(self):
        return [self.cl, '/std:c++latest', '/Zs', '/EHs', '/permissive']

    def get_c_compiler_prefix(self):
        return [self.cl, '/Zs']

    def include(self, inc):
        return '/I' + str(inc)

    def output(self, out):
        return ['/Fo'+out]

    def source(self, src):
        return [src]

    def write_response_file(self, response_file, args):
        for a in args:
            response_file.write((a+'\n').encode('utf-8'))

def find_compiler():
    '''
    Looks for the best available compiler on this platform.
    '''
    cl = shutil.which('cl')
    clang = shutil.which('clang')
    clangpp = shutil.which('clang++')
    if prefer_cl and cl is not None:
        return MsvcCompiler(cl)
    elif clang is not None and clangpp is not None:
        return ClangCompiler(clang, clangpp)
    elif cl is not None:
        return MsvcCompiler(cl)
    else:
        return None

def get_thread_count():
    try:
        n = int(os.environ['CODEQL_THREADS'])
        return n if n>0 else multiprocessing.cpu_count()
    except:
        return multiprocessing.cpu_count()

def is_enabled():
    return os.environ.get('CODEQL_EXTRACTOR_CPP_BUILD_MODE_NONE', 'false') == 'true'
