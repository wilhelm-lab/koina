import torch
import torch.nn as nn
import numpy as np
from depthcharge.encoders import FloatEncoder
from depthcharge.transformers import SpectrumTransformerEncoder

class TransformerModel(nn.Module):
    """
    A Transformer-based model for spectrum prediction.

    Args:
        d_model (int): Dimension of the model embeddings.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        num_classes (int): Number of output classes.
        device (torch.device): Device to run the model on.
        dropout (float): Dropout rate.
        max_seq_length (int): Maximum sequence length.

    Methods:
        attend_mask(tensor): Creates an attention mask for the input tensor.
        forward(mz_array1, intensity_array1, neutral_loss_1, mz_array2, intensity_array2, neutral_loss_2, precursor_information):
            Defines the forward pass of the model.
    """
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, num_classes, device, dropout=0.0, max_seq_length=100):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.nheads = nhead
        self.device = device

        # Initialize the FloatEncoder
        self.float_encoder = FloatEncoder(d_model=d_model, min_wavelength=0.001, max_wavelength=10_000)

        # Transformer Encoder
        self.transformer_encoder = SpectrumTransformerEncoder(d_model=d_model, nhead=self.nheads, dim_feedforward=dim_feedforward, n_layers=num_encoder_layers, dropout=dropout, peak_encoder=True)

        # Transformer Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)

        # Final classification layer
        self.fc_out = nn.Linear(d_model, num_classes)

    def attend_mask(self, tensor):
        """
        Creates an attention mask for the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Attention mask.
        """
        # Create a padding mask where 0 indicates padding
        padding_mask = tensor == 0  # Shape: (batch_size, seq_len)
        batch_size = tensor.shape[0]
        seq_len = tensor.shape[1]

        # Initialize an expanded attention mask with False values
        attention_mask = torch.zeros(batch_size, seq_len + 1, seq_len + 1, dtype=torch.bool)

        # Copy the padding mask to the appropriate positions
        attention_mask[:, 1:, 1:] = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)

        # Expand the mask for the number of attention heads
        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, self.nheads, seq_len + 1, seq_len + 1)

        # Reshape the mask to match the expected input shape for multi-head attention
        attention_mask = attention_mask.reshape(batch_size * self.nheads, seq_len + 1, seq_len + 1)
        return attention_mask.to(self.device)

    def forward(self, mz_array1, intensity_array1, neutral_loss_1, mz_array2, intensity_array2, neutral_loss_2, precursor_information):
        """
        Defines the forward pass of the model.

        Args:
            mz_array1 (torch.Tensor): Mass-to-charge ratio array 1.
            intensity_array1 (torch.Tensor): Intensity array 1.
            neutral_loss_1 (torch.Tensor): Neutral loss array 1.
            mz_array2 (torch.Tensor): Mass-to-charge ratio array 2.
            intensity_array2 (torch.Tensor): Intensity array 2.
            neutral_loss_2 (torch.Tensor): Neutral loss array 2.
            precursor_information (torch.Tensor): Precursor information.

        Returns:
            torch.Tensor: Model output.
        """
        try:
            batch_size = mz_array1.shape[0]
            seq_len = mz_array1.shape[1]

            attention_mask1 = self.attend_mask(mz_array1)
            attention_mask2 = self.attend_mask(mz_array2)
            spectra1_embedding, mem_mask1 = self.transformer_encoder(mz_array1, intensity_array1, mask=attention_mask1)
            spectra2_embedding, mem_mask2 = self.transformer_encoder(mz_array2, intensity_array2, mask=attention_mask2)
            neutral_loss1_embedding, mem_mask_3 = self.transformer_encoder(neutral_loss_1, intensity_array1, mask=attention_mask1)
            neutral_loss2_embedding, mem_mask_4 = self.transformer_encoder(neutral_loss_2, intensity_array2, mask=attention_mask2)

            precursor_embedding = self.float_encoder(precursor_information)

            mem_mask = mem_mask1 & mem_mask2 & mem_mask_3 & mem_mask_4
            spectral_embedding = spectra1_embedding + spectra2_embedding + neutral_loss1_embedding + neutral_loss2_embedding

            tgt_seq_len = precursor_embedding.shape[1]
            mem_mask = mem_mask.unsqueeze(1).expand(batch_size, tgt_seq_len, seq_len + 1)

            # Reshape mem_mask to match the expected input shape for multi-head attention
            mem_mask = mem_mask.unsqueeze(1).expand(batch_size, self.nheads, tgt_seq_len, seq_len + 1)
            mem_mask = mem_mask.reshape(batch_size * self.nheads, tgt_seq_len, seq_len + 1)

            # Transformer Decoder
            decoder_output = self.transformer_decoder(tgt=precursor_embedding, memory=spectral_embedding, memory_mask=mem_mask)

            # Classification layer to get the final probability distribution
            output = self.fc_out(decoder_output[:, 0, :])
            return output
        except Exception as e:
            raise ValueError(f"Error during forward pass: {e}")

def preprocess_example(mz_array1, intensity_array1, mz_array2, intensity_array2, precursor_mass_diff, precursor_mz1, precursor_mz2, max_seq_len=100):
    """
    Preprocesses a single example by padding sequences and converting them to tensors.

    Args:
        mz_array1 (list): List of mass-to-charge ratio values for array 1.
        intensity_array1 (list): List of intensity values for array 1.
        mz_array2 (list): List of mass-to-charge ratio values for array 2.
        intensity_array2 (list): List of intensity values for array 2.
        precursor_mass_diff (float): Precursor mass difference.
        precursor_mz1 (float): Precursor mass-to-charge ratio 1.
        precursor_mz2 (float): Precursor mass-to-charge ratio 2.
        max_seq_len (int): Maximum sequence length for padding.

    Returns:
        tuple: Preprocessed mz_array1, intensity_array1, neutral_loss_1, mz_array2, intensity_array2, neutral_loss_2, and precursor_info as tensors.
    """
    try:
        def pad_sequence(sequence, max_length=max_seq_len):
            if len(sequence) < max_length:
                padding_length = max_length - len(sequence)
                sequence = np.pad(sequence, (0, padding_length), 'constant')
            return sequence[:max_length]

        mz_array1 = pad_sequence(mz_array1)
        intensity_array1 = pad_sequence(intensity_array1)
        neutral_loss_1 = precursor_mz1 - np.array(mz_array1)

        mz_array2 = pad_sequence(mz_array2)
        intensity_array2 = pad_sequence(intensity_array2)
        neutral_loss_2 = precursor_mz2 - np.array(mz_array2)

        mz_array1 = torch.tensor(mz_array1, dtype=torch.float32).unsqueeze(0)
        intensity_array1 = torch.tensor(intensity_array1, dtype=torch.float32).unsqueeze(0)
        neutral_loss_1 = torch.tensor(neutral_loss_1, dtype=torch.float32).unsqueeze(0)

        mz_array2 = torch.tensor(mz_array2, dtype=torch.float32).unsqueeze(0)
        intensity_array2 = torch.tensor(intensity_array2, dtype=torch.float32).unsqueeze(0)
        neutral_loss_2 = torch.tensor(neutral_loss_2, dtype=torch.float32).unsqueeze(0)

        precursor_info = torch.tensor([precursor_mass_diff, precursor_mz1, precursor_mz2], dtype=torch.float32).unsqueeze(0)

        return mz_array1, intensity_array1, neutral_loss_1, mz_array2, intensity_array2, neutral_loss_2, precursor_info
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

def predict_single_modification_site(mz_array1, intensity_array1, mz_array2, intensity_array2, precursor_mass_diff, precursor_mz1, precursor_mz2,modelname = "predict_modification_site.model"):
    """
    Infers the class of a single example using the trained model.

    Args:
        mz_array1 (list): List of mass-to-charge ratio values for array 1.
        intensity_array1 (list): List of intensity values for array 1.
        mz_array2 (list): List of mass-to-charge ratio values for array 2.
        intensity_array2 (list): List of intensity values for array 2.
        precursor_mass_diff (float): Precursor mass difference.
        precursor_mz1 (float): Precursor mass-to-charge ratio 1.
        precursor_mz2 (float): Precursor mass-to-charge ratio 2.

    Returns:
        int: Predicted class.
    """
    try:
        model, device = init_model(modelname)
        model = model.to(device)
        model.eval()

        mz_array1, intensity_array1, neutral_loss_1, mz_array2, intensity_array2, neutral_loss_2, precursor_info = preprocess_example(
            mz_array1, intensity_array1, mz_array2, intensity_array2, precursor_mass_diff, precursor_mz1, precursor_mz2
        )

        mz_array1, intensity_array1, neutral_loss_1, mz_array2, intensity_array2, neutral_loss_2, precursor_info = (
            mz_array1.to(device),
            intensity_array1.to(device),
            neutral_loss_1.to(device),
            mz_array2.to(device),
            intensity_array2.to(device),
            neutral_loss_2.to(device),
            precursor_info.to(device)
        )

        with torch.no_grad():
            output = model(mz_array1, intensity_array1, neutral_loss_1, mz_array2, intensity_array2, neutral_loss_2, precursor_info)
            prediction = torch.argmax(output, axis=1).item()

        return int(prediction == 1)
    except Exception as e:
        raise ValueError(f"Error during inference: {e}")

def init_model(modelname):
    """
    Initializes the model and loads the pre-trained weights.

    Returns:
        tuple: Model and device.
    """
    try:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        # Model instantiation
        d = 128  # Dimension of the embeddings
        nhead = 8  # Number of heads in the multiheadattention models
        num_encoder_layers = 3  # Number of encoder layers
        num_decoder_layers = 3  # Number of decoder layers
        dim_feedforward = 1024  # Dimension of the feedforward network model
        num_classes = 3  # Number of classes
        dropout = 0.2

        model = TransformerModel(d, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, num_classes, device, dropout=dropout)

        # Load the pre-trained model
        model.load_state_dict(torch.load(modelname, map_location=device))
        return model, device
    except Exception as e:
        raise ValueError(f"Error during model initialization: {e}")

# Example usage
if __name__ == "__main__":
    mz_array1 = [100, 200, 300] 
    intensity_array1 = [10, 20, 30]  
    mz_array2 = [150, 250, 350]  
    intensity_array2 = [15, 25, 35]  
    precursor_mass_diff = 20.0  
    precursor_mz1 = 500.0  
    precursor_mz2 = 600.0  

    prediction = predict_single_modification_site(mz_array1, intensity_array1, mz_array2, intensity_array2, precursor_mass_diff, precursor_mz1, precursor_mz2)
    print(f"%%%% TEST %%%% Predicted class: {prediction} %%%%/")
