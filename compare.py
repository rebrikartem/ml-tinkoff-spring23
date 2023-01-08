import argparse
from os import path
import ast
import numpy as np
import multiprocessing

parser = argparse.ArgumentParser(description="Calculate the similarity of python codes")
parser.add_argument("input_file", help="Input file with pairs of py codes")
parser.add_argument("output_file", help="Output file to store the similarity")

args = parser.parse_args()

def remove_docstring(parsed: ast.Module):
    for node in ast.walk(parsed):
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            continue

        if not len(node.body):
            continue

        if not isinstance(node.body[0], ast.Expr):
            continue

        if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
            continue

        node.body = node.body[1:]

def remove_annotations(parsed: ast.Module):
    for node in ast.walk(parsed):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        if not len(node.body):
            continue

        new_args = []
        for arg in node.args.args:
            new_args.append(ast.arg(arg.arg))
        node.args.args = new_args

def normalize_variables(parsed: ast.Module):
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Name):
            continue

        node.id = node.id.replace("_", "").lower()

def sort_imports(module: ast.Module):
    imports = []
    for node in module.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue

        module.body.remove(node)
        imports.append(node)
        
    imports.sort(key=lambda i: ast.unparse(i))
    module.body = imports = module.body

def sort_classes_and_functions(module: ast.Module):
    for node in ast.walk(module):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        classes = []
        functions = []
        for fun in node.body:
            if isinstance(fun, ast.ClassDef):
                fun.name = fun.name.lower().replace("_", "")
                classes.append(fun)

            if isinstance(fun, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if fun.name != "__init__" and fun.name != "__call__":
                    fun.name = fun.name.lower().replace("_", "")
                functions.append(fun)

        for cl in classes:
            node.body.remove(cl)

        for fun in functions:
            node.body.remove(fun)

        classes.sort(key=lambda cl: (len(cl.body), cl.name))
        functions.sort(key=lambda fun: (len(fun.args.args), len(fun.body), fun.name))

        node.body = node.body + classes + functions

def rename_arguments_and_variables(module: ast.Module):
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        old_names_map = {}

        def create_new_name(old_name) -> str:
            nonlocal old_names_map
            new_name = "arg" + str(len(old_names_map))
            old_names_map[old_name] = new_name
            return new_name

        for argument in ast.walk(node.args):
            if isinstance(argument, ast.arg):
                argument.arg = create_new_name(argument.arg)

        for varible in ast.walk(node):
            if isinstance(varible, ast.Name):
                if old_names_map.get(varible.id) is None:
                    varible.id = create_new_name(varible.id)
                else:
                    varible.id = old_names_map.get(varible.id, varible.id)

def calculate_levenshtein_distance(str1: str, str2: str) -> float:
    size_x = len(str1) + 1
    size_y = len(str2) + 1
    matrix = np.zeros((size_x, size_y))
    matrix[:, 0] = np.arange(size_x)
    matrix[0, :] = np.arange(size_y)
    
    for x in range(1, size_x):
        for y in range(1, size_y):
            if str1[x-1] == str2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )
    
    return matrix[size_x - 1, size_y - 1]

def calculate_similarity(str1: str, str2: str) -> float:
    distance = calculate_levenshtein_distance(str1, str2)
    return 1 - distance / max(len(str1), len(str2))

def normalize_code(code: str) -> str:
    module = ast.parse(code)

    remove_docstring(module)
    remove_annotations(module)
    sort_imports(module)
    sort_classes_and_functions(module)
    rename_arguments_and_variables(module)

    normalize_variables(module)
    normalized_code = ast.unparse(module)
    normalized_code = normalized_code.replace("\'", "\"")

    return normalized_code

def compare_two_files(file1: str, file2: str) -> float:

    if path.isfile(file1) and path.isfile(file2):

        with open(file1) as f1:
            code1 = f1.read()
        with open(file2) as f2:
            code2 = f2.read()

        normalized_code1 = normalize_code(code1)
        normalized_code2 = normalize_code(code2)

        return calculate_similarity(normalized_code1, normalized_code2)
        
    else:
        print("Some error occured...")

    return 0

def run_compare(line: str) -> float:
    files = line.split()
    print(files[0], files[1])
    result = compare_two_files(files[0], files[1])
    return result

if __name__ == '__main__':
    input_file = args.input_file
    output_file = args.output_file

    if path.isfile(input_file) and path.isfile(output_file):
        input = open(input_file, 'r')
        lines = input.readlines()
        similarity = []

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            values = pool.map(run_compare, lines)
            similarity = [str(result) + "\n" for result in values]

        output = open(output_file, 'w')
        output.writelines(similarity)
    else:
        print("Some erroe occured...")
