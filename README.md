# Drug-Molecule-Generation-with-VAE

Reference: https://keras.io/examples/generative/molecule_generation/


# Goal:

This example demonstrates how to use a CVAE to generate new molecules that could potentially be candidates for drug discovery.


# Convolutional Variational Autoencoder:

- A generative model that combines the strengths of **convolutional neural networks** and **variational autoencoders** .
- **Variational Autoencoder (VAE)** works as an **unsupervised learning algorithm** that can learn a **latent representation** of data by encoding it into a **probabilistic distribution** and then **reconstructing** back using the convolutional layers which enables the model to **generate new**, **similar data points** .
- The key working principles of a **CVAE** include the incorporation of convolutional layers, which are adept at **capturing spatial hierarchies within data, making them particularly well-suited for image-related tasks.**
- Additionally, CVAEs utilize variational inference, introducing probabilistic elements to the encoding-decoding process. Instead of producing a **fixed** latent representation, a CVAE generates a **probability distribution** in the **latent space** , enabling the model to learn not just a **single** deterministic representation but a range of possible representations for each input. <br>
  - Convolutional Layers: CVAE leverages the power of convolutional layers to efficiently capture spatial hierarchies and local patterns within images which enables the model to recognize features at different scales, providing a robust representation of the input data.
  - Variational Inference: The introduction of variational inference allows CVAE to capture uncertainty in the latent space to generate a probability distribution rather than producing a single deterministic latent representation, providing a richer understanding of the data distribution and enabling the model to explore diverse latent spaces.
  - Reparameterization Trick: It involves sampling from the learned latent distribution during the training process, enabling the model to backpropagate gradients effectively
<br>

- VAEs are a type of generative model that can learn a latent representation of data and use it to generate new samples that share similar properties to the training data.
- In drug discovery, VAEs can be trained on existing drug molecules and then used to generate new molecules with potentially desirable properties.
- A generative model that combines the strengths of **convolutional neural networks** and **variational autoencoders** .


# Data Preprocessing:

The example uses RDKit, a cheminformatics toolkit, to convert SMILES strings (a text format for representing molecules) into numerical representations suitable for the VAE.
SMILES strings are parsed into graphs representing the atoms and bonds in the molecule.

# Convolutional Variational Autoencoder (CVAE):

The model uses convolutional layers to capture spatial **relationships** between atoms in the molecule graph.
The VAE architecture involves:
- Encoder: This part compresses the molecule graph representation into a latent space, capturing the essential features of the molecule.
- Decoder: This part uses the latent representation to reconstruct the original molecule graph or generate new, similar molecules.
- Training: The CVAE is trained on a dataset of SMILES strings representing existing drug molecules.
The training process minimizes the **reconstruction error** (difference between the original and reconstructed molecule) and a **KL divergence** term that encourages diversity in the generated molecules.

# Generating New Molecules:

- After training, the CVAE can be used to generate new SMILES strings by sampling from the latent space and decoding the samples.
<br>

# Conditional Variational Autoencoder (CVAE):

- Goal: Learns a latent representation of data conditioned on specific input information. It can generate new samples based on that **additional information.**
- Functionality: Similar to VAE with an additional step:
  1. Conditional Input: Takes an **extra input** besides the original data. This input can be a label, category, or any additional information that helps guide the generation process.
The encoder and decoder are modified to incorporate this conditional information.
  2. Output: New data samples that not only resemble the training data but also reflect the provided condition. You have more control over the generated data's specific characteristics.
- Applications: Image generation with specific attributes (e.g., generating images of cats with sunglasses), text generation with a specific style or topic, targeted anomaly detection, data augmentation.

<br>
- **ZINC** contains information about a vast number of commercially available compounds. This includes:
1. SMILES Strings: A text format for representing the structure of molecules. SMILES strings are compact and convenient for storing and manipulating molecular information.
2. Molecular Properties: ZINC provides various properties for each compound, including:
  - logP: This is the water-octanol partition coefficient, which measures how well a compound dissolves in water vs. octanol (an organic solvent). It's important for drug discovery as it affects how a drug interacts with the body.
  - SAS (Synthetic Accessibility Score): This score indicates how difficult or easy it is to synthesize the compound in the lab. An ideal drug candidate should be easily synthesizable at a reasonable cost.
  - QED (Qualitative Estimate of Drug-likeness): This score predicts how well a compound resembles known drugs based on various physicochemical properties. A high QED score suggests the compound could be a promising drug candidate.
