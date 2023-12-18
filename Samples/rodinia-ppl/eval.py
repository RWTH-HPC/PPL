import os
import numpy as np

folders = os.listdir('.')
folders = [x for x in folders if not x.startswith('_') and not x.startswith('.') and not x.__contains__('.')]
folders.sort()
def reformat_size(input):
    for i in range(len(input)):
        #input[i] = str("{:.2e}".format(input[i]))
        input[i] = str(input[i] / 1000)
    return input


def string2Int(array):
    for i in range(len(array)):
        array[i] = int(array[i][:-1])
    return array


def reformat_time(input):
    for i in range(len(input)):
        input[i] = str("{:.2e}".format(input[i]/1000))
    return input


mainpath = os.getcwd()

PPL_size = [0]

Parser = [0]
AST_extension = [0]
APT_construction = [0]
APT_time = [0]
Call_time = [0]
Full_time = [0]

AST_size = [0]
APT_size = [0]
Reduction_factor = [0]

with open('Names.txt') as f:
    Names = f.readlines()

with open('Orig_size.txt') as f:
    C_size = f.readlines()

with open('IR_size.txt') as f:
    LLVM_size = f.readlines()

C_size = string2Int(C_size)

LLVM_size = string2Int(LLVM_size)

for path in folders:
        os.chdir(mainpath + '/' + path)
        files = os.listdir('.')
        files = [x for x in files if x.endswith('.py') or x.endswith('.par')]
        for file in files:
            filename = os.fsdecode(file)
            if filename.endswith("Pattern_Nesting_Tree.py"):
                a = 0  # os.system('python ' + filename)
            elif filename.endswith(".par"):
                if path.endswith("-preprocessing"):
                    PPL_size[len(PPL_size) - 1] += os.path.getsize(filename)
                else:
                    PPL_size.append(os.path.getsize(filename))
        if path.endswith("-preprocessing"):
            with open('Parse_Time.txt') as f:
                localParse = f.readlines()
                localParse = string2Int(localParse)
                Parser[len(Parser) - 1] += int(np.mean(localParse).round())
            with open('AST_Extension.txt') as f:
                localASTGen = f.readlines()
                localASTGen = string2Int(localASTGen)
                AST_extension[len(AST_extension) - 1] += int(np.mean(localParse).round())
            with open('AST_Size.txt') as f:
                localASTSize = f.readlines()
                localASTSize = string2Int(localASTSize)
                AST_size[len(AST_size) - 1] += int(np.mean(localParse).round())
            with open('APT_Generation.txt') as f:
                localAPTTime = f.readlines()
                localAPTTime = string2Int(localAPTTime)
                APT_construction[len(APT_construction) - 1] += int(np.mean(localParse).round())
            with open('APT_Size.txt') as f:
                localAPT_Size = f.readlines()
                localAPT_Size = string2Int(localAPT_Size)
                APT_size[len(APT_size) - 1] += int(np.mean(localParse).round())
            with open('APT_Print.txt') as f:
                localAPTPrint = f.readlines()
                localAPTPrint = string2Int(localAPTPrint)
                APT_time[len(APT_time) - 1] += int(np.mean(localParse).round())
            with open('Call_Tree_Print.txt') as f:
                localCallPrint = f.readlines()
                localCallPrint = string2Int(localCallPrint)
                Call_time[len(Call_time) - 1] += int(np.mean(localParse).round())
            with open('Full_Tree_Print.txt') as f:
                localFullPrint = f.readlines()
                localFullPrint = string2Int(localFullPrint)
                Full_time[len(Full_time) - 1] += int(np.mean(localParse).round())

            Reduction_factor[len(Reduction_factor) - 1] = (
                float(AST_size[len(AST_size) - 1] / APT_size[len(APT_size) - 1]))
        else:
            with open('Parse_Time.txt') as f:
                localParse = f.readlines()
                localParse = string2Int(localParse)
                Parser.append(int(np.mean(localParse).round()))
            with open('AST_Extension.txt') as f:
                localASTGen = f.readlines()
                localASTGen = string2Int(localASTGen)
                AST_extension.append(int(np.mean(localASTGen).round()))
            with open('AST_Size.txt') as f:
                localASTSize = f.readlines()
                localASTSize = string2Int(localASTSize)
                AST_size.append(int(np.mean(localASTSize).round()))
            with open('APT_Generation.txt') as f:
                localAPTTime = f.readlines()
                localAPTTime = string2Int(localAPTTime)
                APT_construction.append(int(np.mean(localAPTTime).round()))
            with open('APT_Size.txt') as f:
                localAPT_Size = f.readlines()
                localAPT_Size = string2Int(localAPT_Size)
                APT_size.append(int(np.mean(localAPT_Size).round()))
            with open('APT_Print.txt') as f:
                localAPTPrint = f.readlines()
                localAPTPrint = string2Int(localAPTPrint)
                APT_time.append(int(np.mean(localAPTPrint).round()))
            with open('Call_Tree_Print.txt') as f:
                localCallPrint = f.readlines()
                localCallPrint = string2Int(localCallPrint)
                Call_time.append(int(np.mean(localCallPrint).round()))
            with open('Full_Tree_Print.txt') as f:
                localFullPrint = f.readlines()
                localFullPrint = string2Int(localFullPrint)
                Full_time.append(int(np.mean(localFullPrint).round()))
            Reduction_factor.append((float(AST_size[len(AST_size) - 1] / APT_size[len(APT_size) - 1])))
        os.chdir(mainpath)

print(Names)
print(C_size)
print(LLVM_size)

print(PPL_size)
print(Parser)
print(AST_extension)
print(APT_construction)
print(Call_time)
print(Full_time)
print(APT_time)
print(AST_size)
print(APT_size)
print(Reduction_factor)

PPL_size = reformat_size(PPL_size)
AST_size = reformat_size(AST_size)
APT_size = reformat_size(APT_size)
C_size = reformat_size(C_size)
LLVM_size = reformat_size(LLVM_size)

AST_Gen = Parser
for i in range(len(AST_Gen)):
    AST_Gen[i] += AST_extension[i]

AST_Gen = reformat_time(AST_Gen)
APT_time = reformat_time(APT_time)
APT_construction = reformat_time(APT_construction)
Full_time = reformat_time(Full_time)

if os.path.exists("Values.txt"):
    os.remove("Values.txt")
else:
    print("Can not delete the file as it doesn't exists")
f = open("Values.txt", "x")

f.write(
    "\\textbf{Benchmark} & \\textbf{Parse Time} & \\textbf{APT Gen.} & \\textbf{APT Print} & \\textbf{Full Print}\\\\\n")
f.write("\\hline\n")
for x in range(len(Names)):
    f.write("\\textbf{" + Names[x][:-1] + "}\t&" + str(AST_Gen[x]) + "\t&" + str(
        APT_construction[x]) + "\t&" + str(APT_time[x]) + "\t&" + str(Full_time[x]) + "\\\\\n")

if os.path.exists("Sizes.txt"):
    os.remove("Sizes.txt")
else:
    print("Can not delete the file as it doesn't exists")

f = open("Sizes.txt", "x")
f.write("\\textbf{Benchmark} & \\textbf{C} & \\textbf{PPL} & \\textbf{LLVM} & \\textbf{APT} \\\\\n")
f.write("\\hline\n")
for x in range(len(Names)):
    f.write("\\textbf{" + Names[x][:-1] + "}\t&" + C_size[x] + "\t&" + PPL_size[
        x] + "\t&" + LLVM_size[x] + "\t&" + APT_size[x] + "\\\\\n")
