from config import get_model_name, load_config
from datasets import load_dataset
from trainers import SuperTrainer, VAEModule, PixelCNNModule
from loss import VAELoss
from plotting import plot_samples_with_reconstruction_and_indices, generate_indices_and_reconstruct
from plotting import plot_samples_with_reconstruction, plot_latent_images
def train_and_evaluate(config_path, verbose=False):
    """Fully trains and evaluates a model for a given config"""

    print("***"*20)

    print("Starting training for config: ", "configs/"+config_path)

    config = load_config(config_path)

    model_name = get_model_name(config)

    print("Model name: ", model_name)

    print("***"*20)

    train_loader, test_loader, val_loader = load_dataset(config["data_params"])

    model = VAEModule(config['model_params'], model_name=config['model_name'])

    trainer = SuperTrainer(**config['trainer_params'], model_name=model_name)
    trainer.fit(model, train_loader, val_loader)
    #saves model checkpoint
    trainer.save_model_checkpoint()


    model.eval()
    # check if loss is instance of VAELoss
    if isinstance(model.model.loss, VAELoss):
        print("Training done")

        plot_samples_with_reconstruction(model, next(iter(test_loader)), save_name = model_name)
        plot_latent_images(model, save_name = model_name)

        print("All done")
        print("----"*20)

    else:
        plot_samples_with_reconstruction_and_indices(model, next(iter(test_loader)), save_name=model_name)

        print("1st stage training done")

        # 2nd stage training
        pixel_cnn_model_name = "Pixel_CNN_" + model_name
        model = PixelCNNModule(vae_name=model_name)

        trainer = SuperTrainer(**config['trainer_params'], model_name = pixel_cnn_model_name)
        trainer.fit(model, train_loader, val_loader)
        #save
        trainer.save_model_checkpoint()

        print("2nd stage training done")

        generate_indices_and_reconstruct(model, save_name=pixel_cnn_model_name)


        print("All done")
        print("----"*20)









