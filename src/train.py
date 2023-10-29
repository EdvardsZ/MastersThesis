from config import get_model_name
from datasets import load_dataset
from trainers import VAEModule, PixelCNNModule
from loss import VAELoss
from plotting import plot_samples_with_reconstruction_and_indices, generate_indices_and_reconstruct
from plotting import plot_samples_with_reconstruction, plot_latent_images, plot_stage_one_results
from lightning_extensions import ExtendedTrainer
def train_and_evaluate(config, verbose=False):
    """Fully trains and evaluates a model for a given config"""

    print("***"*20)

    model_name = get_model_name(config)

    print("Starting training for model: ", model_name)

    print("***"*20)

    train_loader, test_loader, val_loader = load_dataset(config["data_params"])

    model = VAEModule(config['model_params'], model_name=config['model_name'])

    trainer = ExtendedTrainer(project_name="MultiTaskVariationalAutoecnoders_test", **config['trainer_params'], model_name=model_name, )
    trainer.fit(model, train_loader, val_loader)
    #saves model checkpoint
    trainer.save_model_checkpoint()


    model.eval()
    print("1st stage training done")
    plot_stage_one_results(model_name, test_loader)

    if isinstance(model.model.loss, VAELoss):

        print("Starting 2nd stage training")

        # 2nd stage training
        pixel_cnn_model_name = "Pixel_CNN_" + model_name
        model = PixelCNNModule(vae_name=model_name)

        trainer = ExtendedTrainer(project_name="MultiTaskVariationalAutoecnoders_test",**config['trainer_params'], model_name = pixel_cnn_model_name)
        trainer.fit(model, train_loader, val_loader)
        #save
        trainer.save_model_checkpoint()

        print("2nd stage training done")

        generate_indices_and_reconstruct(model, save_name=pixel_cnn_model_name)


    print("All done")
    print("----"*20)









