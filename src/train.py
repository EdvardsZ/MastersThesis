from config_loader import get_model_name
from datasets import load_dataset
from modules import VAEModule
from lightning_extensions import ExtendedTrainer
from loss import VQLoss
from plotting import plot_stage_one_results
import torch
import os
from crossval_result_loader import check_if_done_already, check_if_pending, mark_as_pending, remove_pending
    
    
def train_and_evaluate(config: dict, project_name: str, device: int, cross_validation = False):
    """Fully trains and evaluates a model for a given config"""

    print("***"*20)

    model_name = get_model_name(config)
    
    
    if cross_validation:
        if check_if_done_already(model_name, project_name):
            print("Job is already done(Results already exist)")
            print("Training skipped")
            print("***"*20)
            return
        if check_if_pending(model_name, project_name):
            print("Job is pending")
            print("Training skipped")
            print("***"*20)
            return
        mark_as_pending(model_name, project_name, job_nr = device)

    print("Starting training for model: ", model_name)

    print("***"*20)

    train_loader, test_loader, val_loader, image_shape = load_dataset(config["data_params"])

    model = VAEModule(config['model_params'], model_name=config['model_name'], image_shape=image_shape)

    trainer = ExtendedTrainer(project_name="MTVAEs_05.01-cross-val", **config['trainer_params'], model_name=model_name, devices = [device],refresh_rate = 60)

    if cross_validation:
        trainer.cross_validate(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)

        #saves model checkpoint
        trainer.save_model_checkpoint()


        model.eval()
        plot_stage_one_results(model, val_loader, save_name=model_name)
        
    print("1st stage training done")



    # TODO second stage training
    # if isinstance(model.model.loss, VQLoss):

    #     print("Starting 2nd stage training")

    #     # 2nd stage training
    #     pixel_cnn_model_name = "Pixel_CNN_" + model_name
    #     model = PixelCNNModule(vae_name=model_name)

    #     trainer = ExtendedTrainer(project_name="MultiTaskVariationalAutoecnoders_test",**config['trainer_params'], model_name = pixel_cnn_model_name)
    #     trainer.fit(model, train_loader, val_loader)
    #     #save
    #     trainer.save_model_checkpoint()

    #     print("2nd stage training done")

    #     generate_indices_and_reconstruct(model, save_name=pixel_cnn_model_name)


    print("All done")
    print("----"*20)
    if cross_validation:
        remove_pending(model_name, project_name)