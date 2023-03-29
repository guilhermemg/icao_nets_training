
import matplotlib.pyplot as plt

from src.base.experiment.dataset.dataset import Dataset

class ModelTrainVisualizer:
    def __init__(self, config_interp):
        self.config_interp = config_interp

    
    def visualize_history(self, history):
        f = None
        if not self.config_interp.is_mtl_model:
            f,ax = plt.subplots(1,2, figsize=(10,5))
            f.suptitle(f'-----{self.config_interp.base_model.name}-----')

            ax[0].plot(history.history['accuracy'])
            ax[0].plot(history.history['val_accuracy'])
            ax[0].set_title('Model Accuracy')
            ax[0].set_ylabel('accuracy')
            ax[0].set_xlabel('epoch')
            ax[0].legend(['train', 'validation'])

            ax[1].plot(history.history['loss'])
            ax[1].plot(history.history['val_loss'])
            ax[1].set_title('Model Loss')
            ax[1].set_ylabel('loss')
            ax[1].set_xlabel('epoch')
            ax[1].legend(['train', 'validation'])

        else:
            f,ax = plt.subplots(2,2, figsize=(20,25))
            f.suptitle(f'-----{self.config_interp.base_model.name}-----')

            for task in self.config_interp.tasks:
                ax[0][0].plot(history.history[f'{task.value}_accuracy'])
                ax[0][1].plot(history.history[f'val_{task.value}_accuracy'])

                ax[1][0].plot(history.history[f'{task.value}_loss'])
                ax[1][1].plot(history.history[f'val_{task.value}_loss'])

            ax[1][0].plot(history.history['loss'], color='red', linewidth=2.0) # total loss

            ax[0][0].set_title('Model Accuracy - Train')
            ax[0][1].set_title('Model Accuracy - Validation')

            ax[0][0].set_ylabel('accuracy')
            ax[0][1].set_ylabel('accuracy')
            ax[0][0].set_xlabel('epoch')
            ax[0][1].set_xlabel('epoch')

            ax[0][0].set_ylim([0,1.1])
            ax[0][1].set_ylim([0,1.1])

            ax[1][0].set_title('Model Loss - Train')
            ax[1][1].set_title('Model Loss - Validation')

            ax[1][0].set_ylabel('loss')
            ax[1][1].set_ylabel('loss')

            ax[1][0].set_xlabel('epoch')
            ax[1][1].set_xlabel('epoch')

            ax[1][0].set_ylim([0,1.5])
            ax[1][1].set_ylim([0,1.5])

            legends = self.config_interp.dataset.value['target_cols']

            ax[0][0].legend(legends, ncol=4)
            ax[0][1].legend(legends, ncol=4)
            ax[1][0].legend(legends, ncol=4)
            ax[1][1].legend(legends, ncol=4)
        
        plt.show()

        return f