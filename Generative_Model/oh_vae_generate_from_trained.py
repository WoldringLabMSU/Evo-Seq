import pandas as pd
import torch
from vae_oh_CNN import ProteinVAE
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os


def one_hot_decode(encoded_seq, amino_acids="ACDEFGHIKLMNPQRSTVWY-"):
    """
    /mnt/home/mardikor/ASR-GEN/ASR_Preprocessing/PK2_all/sample_fasta.pyConvert the one-hot encoded sequence tensor back to a string of amino acids.
    :param encoded_seq: Tensor of shape [1, 21, sequence_length]
    :param amino_acids: A string containing the amino acids in the order they were one-hot encoded.
    :return: The decoded amino acid sequence.
    """

    print(encoded_seq.shape)
    # Get the indices of max values along dimension 1 (amino acid channels)
    decoded_indices = torch.argmax(encoded_seq, dim=2)
    print(decoded_indices)
    # Convert the indices to amino acids
    sequence = ''.join([amino_acids[idx] for idx in decoded_indices[0]])
    
    return sequence


def generate_sequences(model_path, num_samples=1000):
    # Load the VAE model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = ProteinVAE().to(device)
    vae.load_state_dict(torch.load(model_path))
    vae.eval()

    generated_sequences = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Sample from latent space
            z = torch.randn(1, vae.encoder.z_mean.out_features).to(device)
            
            # Decode the sample
            decoded_probabilities = vae.decoder(z)
           
            # Convert the probabilities tensor into amino acid indices
            decoded_indices = torch.argmax(decoded_probabilities, dim=1)
            
            # Convert the indices tensor into an actual sequence
            sequence = one_hot_decode(decoded_probabilities.cpu())
            generated_sequences.append(sequence)
          
    return generated_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained/fine-tuned model.')
    parser.add_argument('--csv_path', type=str, required=False, help='Path to save output sequences in csv.')
    args = parser.parse_args()

    model_path = args.model_path
    
    if args.csv_path is not None:
        csv_path = args.csv_path
    else:
        csv_path = f'{os.path.splitext(os.path.basename(model_path))[0]}.csv'
    
    sequences = generate_sequences(model_path, num_samples=1000)
    sequences= pd.DataFrame(sequences).to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()
