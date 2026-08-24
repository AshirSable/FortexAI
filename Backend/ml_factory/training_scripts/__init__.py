from ml_factory.training_scripts.scripts_ae import (
        base_ae as base_autoencoder_training,
        base_ae_attack as base_autoencoder_attack_training,
        base_ae_attack_contrastive as base_auto_encoder_contrastive_training
        )

from ml_factory.training_scripts.scripts_normality_ae import (
        base as base_auto_mixture_training,
        base_attack as base_auto_mixture_attack_training,
        base_attack_diversity as base_auto_mixture_diversity_training,
        base_attack_router as base_auto_mixture_router_training
        )


if __name__ == '__main__':
    base_ae_training = [base_autoencoder_training, base_autoencoder_attack_training, base_auto_encoder_contrastive_training]
    mixture_ae_training = [
            base_auto_mixture_training,
            base_auto_mixture_attack_training,
            base_auto_mixture_diversity_training,
            base_auto_mixture_router_training
    ]


    print('Base AE training')
    print('*'*10)
    for bt in base_ae_training:

        print(f"Starting training: {bt.__name__}")

        bt()

        print(f"Training Ended")



    print('*'*10)
    print('Mixture AE training')
    print('*'*10)
    for bt_m in mixture_ae_training:

        print(f"Starting training: {bt_m.__name__}")

        bt_m()

        print(f"Training Ended")
