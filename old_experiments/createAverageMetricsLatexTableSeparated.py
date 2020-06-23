import pickle

fileSVM = "./filesFolds-SVM/output/evaluationMean.p"
fileSVMbigrams = "./filesFolds-SVMbigrams/output/evaluationMean.p"
fileSVMseparated = "./filesFolds-SVMbigramsSeparated/output/evaluationMean.p"
fileLSTMnoGloVe = "./filesFolds-LSTMnoGloVe/output/evaluationMean.p"
fileLSTMconvolutional = "./filesFolds-LSTMconvolutional2/output/evaluationMean.p"
fileLSTMbidirectional = "./filesFolds-LSTMbidirectional5e/output/evaluationMean.p"
fileAladar = "./metricsOld.p"

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

metrics = [pickle.load(open(fileSVM, 'rb')), pickle.load(open(fileSVMbigrams, 'rb')), pickle.load(open(fileSVMseparated, 'rb')), pickle.load(open(fileLSTMnoGloVe, 'rb')), pickle.load(open(fileLSTMconvolutional, 'rb')), pickle.load(open(fileLSTMbidirectional, 'rb'))]
metricsAladar = pickle.load(open(fileAladar, 'rb'))

if useSingleFile:
    out = open(fileOutSingle, 'wt')
    
    out.write("\\documentclass{article}\n")
    out.write("\\usepackage{multirow}\n")
    out.write("\\usepackage{rotating}\n")
    out.write("\\newcommand{\\aladar}{\\textbf{\\emph{Aladar}}}\n")
    out.write("\\newcommand{\\svm}{\\textbf{\\emph{TF-IDF}}}\n")
    out.write("\\newcommand{\\svmb}{\\textbf{\\emph{TF-IDF-b}}}\n")
    out.write("\\newcommand{\\svms}{\\textbf{\\emph{TF-IDF-s}}}\n")
    out.write("\\newcommand{\\lstmng}{\\textbf{\\emph{no-GLOVE}}}\n")
    out.write("\\newcommand{\\lstmc}{\\textbf{\\emph{CONV}}}\n")
    out.write("\\newcommand{\\lstmb}{\\textbf{\\emph{deep-LSTM}}}\n")
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
        
    out.write("        \\begin{tabular}{|l|l|c|r@{$\;\pm\;$}l|r@{$\;\pm\;$}l|r@{$\;\pm\;$}l|r@{$\;\pm\;$}l|r@{$\;\pm\;$}l|r@{$\;\pm\;$}l|}\n")
    out.write("            \\hline\n")
    out.write("            \\multicolumn{2}{|c|}{} & \\aladar & \\multicolumn{2}{c|}{\\svm} & \\multicolumn{2}{c|}{\\svmb} & \\multicolumn{2}{c|}{\\svms} & \\multicolumn{2}{c|}{\\lstmng} & \\multicolumn{2}{c|}{\\lstmc} & \\multicolumn{2}{c|}{\\lstmb}\\\\\n")
    for l in simpleLabels:
        out.write("            \\hline\n")
        out.write(("            \\multicolumn{{2}}{{|l|}}{{\\textbf{{{}}}}} & $"+formatStrAladar(l)+"$").format(l, formatAladar(l, metricsAladar[t][l])))
        for m in metrics:
            out.write((" & $"+formatStr+"$ & $"+formatStr+"$").format(formatFun(m[t][l]['mean']), formatFun(m[t][l]['sd'])))
        out.write(" \\\\\n")

    for l in avgLabels:
        out.write("            \\hline\n")
        out.write(("            \\multirow{{2}}{{*}}{{\\textbf{{{}}}}} & \\textbf{{{}}} & $"+formatStrAladar(l)+"$").format(avgLabelsNames[l], avgNames['macro'], formatAladar(l, metricsAladar[t][l]['macro'])))
        for m in metrics:
            out.write((" & $"+formatStr+"$ & $"+formatStr+"$").format(formatFun(m[t][l]['macro']['mean']), formatFun(m[t][l]['macro']['sd'])))
        out.write(" \\\\\n")
        out.write("            \\cline{2-15}")
        out.write(("& \\textbf{{{}}} & $"+formatStrAladar(l)+"$").format(avgNames['weighted'], formatAladar(l, metricsAladar[t][l]['weighted'])))
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

