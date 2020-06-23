import pickle

fileModel = {
    'Aladar' : "./metricsOld.p",
    'SVM' : "./filesFolds-SVM/output/evaluationMean.p",
    'SVMbigrams' : "./filesFolds-SVMbigrams/output/evaluationMean.p",
    'LSTMnoGloVe' : "./filesFolds-LSTMnoGloVe/output/evaluationMean.p",
    'LSTMconvolutional' : "./filesFolds-LSTMconvolutional2/output/evaluationMean.p",
    'LSTMbidirectional' : "./filesFolds-LSTMbidirectional5e/output/evaluationMean.p",
    'LSTMbidirectionalNoNegation' : "./filesFolds-LSTMbidirectional5eNoNegation/output/evaluationMean.p",
}

modelCommand = {
    'Aladar' : ("\\aladar", "\\textbf{\\emph{Ala}}"),
    'SVM' : ("\\svm", "\\textbf{\\emph{T-SVM}}"),
    'SVMbigrams' : ("\\svmb", "\\textbf{\\emph{T2-SVM}}"),
    'LSTMnoGloVe' : ("\\lstmng", "\\textbf{\\emph{T2-2LSTM}}"),
    'LSTMconvolutional' : ("\\lstmc", "\\textbf{\\emph{G-CLSTM}}"),
    'LSTMbidirectional' : ("\\lstmb", "\\textbf{\\emph{G-2LSTM}}"),
    'LSTMbidirectionalNoNegation' : ("\\lstmbnn", "\\textbf{\\emph{G-2LSTMnn}}"),
}
    

#addAladar = True
#addModels = ['SVM', 'SVMbigrams', 'LSTMnoGloVe', 'LSTMconvolutional', 'LSTMbidirectional']
addAladar = False
addModels = ['LSTMbidirectional', 'LSTMbidirectionalNoNegation']

#useSingleFile = False
useSingleFile = True

fileOutSingle = "./tabs/tabs.tex"
fileOut = {
    'sede1' : "./tabs/site.tex",
    'sede12' : "./tabs/fullSite.tex",
    'morfo1' : "./tabs/type.tex",
    'morfo2' : "./tabs/behaviour.tex",
}

tasks = ['sede1', 'sede12', 'morfo1', 'morfo2']
#tasks = ['sede12']
simpleLabels = ['accuracy', 'kappa', 'MAPs', 'MAPc']
avgLabels = ['precision', 'recall', 'f1score']
aladarLabels = ['accuracy', 'kappa', 'precision', 'recall', 'f1score']

avgLabelsNames={
    'precision':"pre.",
    'recall':"rec.",
    'f1score':"f1s.",
}
avgNames={
    'macro':"ma.",
    'weighted':"we.",
}

titleCommand = {
    'sede1' : 'site',
    'sede12' : 'fullSite',
    'morfo1' : 'type',
    'morfo2' : 'behaviour',
}

usePercent = True
if usePercent:
    formatStr = "{:.1f}"
    formatFun = lambda x:x*100
else:
    formatStr = "{:.3f}"
    formatFun = lambda x:x

formatStrAladar = lambda l : formatStr if l in aladarLabels else "{}"
formatAladar = lambda l,x : formatFun(x) if l in aladarLabels else "N/A" 

metricsAladar = pickle.load(open(fileModel['Aladar'], 'rb'))
metrics = [ pickle.load(open(fileModel[mod], 'rb')) for mod in addModels ]

clineEnd = 2 * len(addModels) + 2
if addAladar:
    clineEnd += 1

if useSingleFile:
    out = open(fileOutSingle, 'wt')
    
    out.write("\\documentclass{article}\n")
    out.write("\\usepackage{multirow}\n")
    out.write("\\usepackage{rotating}\n")
    for com, val in modelCommand.values():
        out.write("\\newcommand{{{}}}{{{}}}\n".format(com, val))
    out.write("\\newcommand{\\site}{site}\n")
    out.write("\\newcommand{\\fullSite}{full site}\n")
    out.write("\\newcommand{\\type}{type}\n")
    out.write("\\newcommand{\\behaviour}{behaviour}\n")
    out.write("\\begin{document}\n")
    out.write("\n")

for t in tasks:
    if not useSingleFile:
        out = open(fileOut[t], 'wt')
    else:
        out.write("    \\begin{sidewaystable}\n")
        out.write("        \\centering\n")
        
    out.write("        \\begin{tabular}{|l|l")
    if addAladar:
        out.write("|c")
    for _ in addModels:
        out.write("|r@{$\;\pm\;$}l")
    out.write("|}\n")
    out.write("            \\hline\n")
    out.write("            \\multicolumn{2}{|c|}{}")
    if addAladar:
        out.write(" & \\aladar")
    for mod in addModels:
        out.write(" & \\multicolumn{{2}}{{c|}}{{{}}}".format(modelCommand[mod][0]))
    out.write("\\\\\n")
    for l in simpleLabels:
        out.write("            \\hline\n")
        out.write("            \\multicolumn{{2}}{{|l|}}{{\\textbf{{{}}}}}".format(l))
        if addAladar:
            out.write((" & $"+formatStrAladar(l)+"$").format(formatAladar(l, metricsAladar[t][l])))
        for m in metrics:
            out.write((" & $"+formatStr+"$ & $"+formatStr+"$").format(formatFun(m[t][l]['mean']), formatFun(m[t][l]['sd'])))
        out.write(" \\\\\n")

    for l in avgLabels:
        out.write("            \\hline\n")
        out.write("            \\multirow{{2}}{{*}}{{\\textbf{{{}}}}} & \\textbf{{{}}}".format(avgLabelsNames[l], avgNames['macro']))
        if addAladar:
            out.write((" & $"+formatStrAladar(l)+"$").format(formatAladar(l, metricsAladar[t][l]['macro'])))            
        for m in metrics:
            out.write((" & $"+formatStr+"$ & $"+formatStr+"$").format(formatFun(m[t][l]['macro']['mean']), formatFun(m[t][l]['macro']['sd'])))
        out.write(" \\\\\n")
        out.write("            \\cline{{2-{}}}".format(str(clineEnd)))
        out.write("& \\textbf{{{}}}".format(avgNames['weighted']))
        if addAladar:
            out.write((" & $"+formatStrAladar(l)+"$").format(formatAladar(l, metricsAladar[t][l]['weighted'])))
        for m in metrics:
            out.write((" & $"+formatStr+"$ & $"+formatStr+"$").format(formatFun(m[t][l]['weighted']['mean']), formatFun(m[t][l]['weighted']['sd'])))
        out.write(" \\\\\n")
    out.write("            \\hline\n")
    out.write("        \\end{tabular}\n")

    if useSingleFile:
        out.write("        \\caption{{Results for \\{} task.}}\n".format(titleCommand[t]))
        out.write("    \\label{{tab:results{}}}\n".format(titleCommand[t]))
        out.write("    \\end{sidewaystable}\n")
    else:
        out.close()

if useSingleFile:
    out.write("\n")
    out.write("\\end{document}\n")

    out.close()

