# imports
import matplotlib.pyplot as plt



def save_training_history_plot(history, model_name, input_shape):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('train/logs/Loss'+ model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+ '.png', format='png')
    return


