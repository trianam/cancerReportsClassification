import pickle
import matplotlib.pyplot as plt

def plotTraining(task, fold, fileBase, numTrainings):
    modelsEpochs = []

    histories = []
    for i in range(numTrainings):
        histories.append(pickle.load(open(fileBase+"/history/"+str(fold)+"/historyCat"+task.capitalize()+"-"+str(i)+".p", 'rb')))

    evaluations = []
    for i in range(numTrainings):
        evaluations.append(pickle.load(open(fileBase+"/output/"+str(fold)+"/evaluation"+task.capitalize()+"-"+str(i)+".p", 'rb')))

    allAcc = []
    allValAcc = []
    allLoss = []
    allValLoss = []
    for history in histories:
        allAcc += history['acc']
        allValAcc += history['val_acc']
        allLoss += history['loss']
        allValLoss += history['val_loss']
        modelsEpochs.append(len(allAcc))

    # metrics on test
    for x in modelsEpochs:
        plt.axvline(x, color='k')
    plt.plot(modelsEpochs, [ev['accuracy'] for ev in evaluations], label='accuracy')
    plt.plot(modelsEpochs, [ev['MAPs'] for ev in evaluations], label='MAPs')
    plt.plot(modelsEpochs, [ev['MAPc'] for ev in evaluations], label='MAPc')
    plt.plot(modelsEpochs, [ev['kappa'] for ev in evaluations], label='kappa')
    plt.title('metrics on test')
    #plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim([0,1])
    plt.show()

    # summarize history for accuracy
    for x in modelsEpochs:
        plt.axvline(x, color='k')
    plt.plot(range(1, len(allAcc)+1), allAcc, label='train')
    plt.plot(range(1, len(allValAcc)+1), allValAcc, label='valid')
    #plt.plot(modelsEpochs, [ev['accuracy'] for ev in evaluations], label='test')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim([0,1])
    plt.show()
    # summarize history for loss
    for x in modelsEpochs:
        plt.axvline(x, color='k')
    plt.plot(range(1, len(allAcc)+1), allLoss, label='train')
    plt.plot(range(1, len(allValAcc)+1), allValLoss, label='valid')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.grid()
    plt.ylim([0,3])
    plt.show()
