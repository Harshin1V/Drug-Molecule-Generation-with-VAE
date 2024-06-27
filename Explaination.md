```
def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule
```
- The above code defines a function molecule_from_smiles(smiles) that takes a **SMILES string (text representation of a molecule) and returns a sanitized and stereochemistry-assigned RDKit molecule object** . 
Here's a breakdown of the steps:

1. Initial Molecule Creation (without sanitization)
- molecule = Chem.MolFromSmiles(smiles, sanitize=False)
- This line uses the Chem.MolFromSmiles function from the RDKit library to create a molecule object from the provided SMILES string.
Important: The sanitize=False argument is used here. Sanitization refers to the process of fixing potential issues in the molecule structure represented by the SMILES string. Disabling sanitization at this stage allows flexibility to handle potential errors later.
2. Error Handling and Selective Sanitization:
- flag = Chem.SanitizeMol(molecule, catchErrors=True)
- if flag != Chem.SanitizeFlags.SANITIZE_NONE:
   - Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
- Chem.SanitizeMol(molecule, catchErrors=True) attempts to sanitize the molecule while catching any errors that might occur during the process.
- The flag variable stores a bitmask indicating the specific sanitization issues encountered.
- If flag is not Chem.SanitizeFlags.SANITIZE_NONE, it means there were issues.
- The code then performs a more targeted sanitization using Chem.SanitizeMol again.
- sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag excludes the specific sanitization operations that caused errors previously (using bitwise XOR). This helps avoid repetitive unsuccessful attempts at specific types of fixes.
3. Assigning Stereochemistry:
- Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
- Chem.AssignStereochemistry assigns explicit stereochemistry information (e.g., cis/trans bonds, chiral centers) to the molecule based on available information and heuristics.
- cleanIt=True removes any existing stereochemistry information before assigning new values.
- force=True forces stereochemistry assignment even if there's not enough information to confidently determine it.
4. Returning the Molecule:
- The function returns the final molecule object, which should be a sanitized RDKit molecule with assigned stereochemistry.
- Overall, it aims to create a robust function for handling SMILES strings and building RDKit molecule objects, even if the initial input might have minor structural issues.


```
print(f"SMILES:\t{df.smiles[100]}\nlogP:\t{df.logP[100]}\nqed:\t{df.qed[100]}")
molecule = molecule_from_smiles(df.iloc[29].smiles)
print("Molecule:")
molecule
```

- This line uses f-strings (formatted string literals) for printing information about a specific molecule in the DataFrame.
```
df.smiles[100] accesses the SMILES string of the molecule at index 100 in the smiles column.
df.logP[100] and df.qed[100] access the corresponding logP and qed values for the same molecule (assuming these columns exist in the DataFrame).
```
- SMILES:\t (molecule structure in SMILES format)
- logP:\t (water-octanol partition coefficient)
- qed:\t (qualitative estimate of drug-likeness)

```
SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", \
                  "P", "Cl", "Br"]'             
bond_mapping = {
    "SINGLE": 0,
    0: Chem.BondType.SINGLE,
    "DOUBLE": 1,
    1: Chem.BondType.DOUBLE,
    "TRIPLE": 2,
    2: Chem.BondType.TRIPLE,
    "AROMATIC": 3,
    3: Chem.BondType.AROMATIC,
}
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)

MAX_MOLSIZE = max(df['smiles'].str.len())
SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
atom_mapping = dict(SMILE_to_index)
atom_mapping.update(index_to_SMILE)

print("Max molecule size: {}".format(MAX_MOLSIZE))
print("Character set Length: {}".format(len(SMILE_CHARSET)))
```

1. Defining SMILES Character Set:
SMILE_CHARSET is a string defining the allowed characters in SMILES strings. It includes standard elements like Carbon (C), Oxygen (O), etc. and bond types like SINGLE, DOUBLE, etc. (written as strings for now).
The ast.literal_eval(SMILE_CHARSET) line (assuming ast is imported) converts the string into a list containing the individual characters and bond type labels. This creates a more usable format.
2. Bond Type Mapping:
bond_mapping is a dictionary that maps string representations of bond types ("SINGLE", "DOUBLE", ...) to their corresponding RDKit bond type objects (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, etc.). This allows easy conversion between string labels and the RDKit representation used for molecule construction.
3. Maximum Molecule Size:
MAX_MOLSIZE is calculated using max(df['smiles'].str.len()). This finds the length (number of characters) of the longest SMILES string in the smiles column of the DataFrame (df). This gives an idea of the maximum molecule complexity in the data.
4. Character and Index Mappings:
Two dictionaries, SMILE_to_index and index_to_SMILE, are created using list comprehension. They map between characters/bond types in the SMILES string and their corresponding integer indices. This allows for efficient numerical representation of the SMILES sequences for potential processing with deep learning models.
Finally, atom_mapping is created by combining SMILE_to_index and updating it with the inverse mapping (index_to_SMILE) using .update(). This dictionary allows for conversion between characters/bond types and their indices in both directions.
5. Printing Information:
The code prints the maximum molecule size and the length of the character set (number of allowed characters and bond types).
Overall, this code snippet prepares the data for processing SMILES strings. It defines the allowed characters, maps bond type labels to RDKit objects, calculates the maximum molecule size, and creates dictionaries for efficient conversion between SMILES characters/bond types and their numerical representations.
```
BATCH_SIZE = 32
EPOCHS =10

VAE_LR = 5e-4
NUM_ATOMS = 120 # Maximum number of atoms

ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
BOND_DIM = 4 + 1  # Number of bond types
LATENT_DIM = 435  # Size of the latent space
```

- Training Parameters:

    - BATCH_SIZE: This is the number of SMILES strings processed by the VAE model in a single training step. A common choice, 32 in this case, provides a balance between computational efficiency and training stability.
EPOCHS: This defines the number of times the entire dataset is passed through the VAE model during training. 10 epochs is a typical starting point, but the optimal number might be determined through validation and monitoring training progress.
- Model Architecture Parameters:

    - NUM_ATOMS: This specifies the maximum number of atoms allowed in a molecule represented by a SMILES string. This value should be chosen based on the dataset and the complexity of the molecules it contains.
    - ATOM_DIM: This is the embedding dimension (number of features) used to represent each atom type in the SMILES string. The value is derived from the length of SMILE_CHARSET, indicating that each atom type will be mapped to a vector of this size.
    - BOND_DIM: This defines the embedding dimension for representing bond types. Here, it's set to 4 + 1. The 4 might represent features like bond directionality or aromaticity, and the 1 could indicate the presence/absence of a bond.
    - LATENT_DIM: This specifies the size of the latent space in the VAE model. This is the dimensionality of the compressed representation of the molecule learned by the VAE. A higher value allows capturing more complex information about the molecules, but it also increases model complexity.
    - Overall, these hyperparameters define how the VAE model will process and learn from the SMILES data. Adjusting these values can influence the training process, the quality of the learned representations, and the performance of the VAE in generating new molecules.

- These are likely just a few of the hyperparameters involved in the complete VAE model. Other parameters related to the network architecture, optimizer configuration, and learning rate scheduling might also be present.
Choosing the optimal hyperparameters often requires experimentation and evaluation based on the specific dataset and desired outcome.


```
def smiles_to_graph(smiles):
    '''
    Reference: https://keras.io/examples/generative/wgan-graphs/
    '''
    # Converts SMILES to molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Initialize adjacency and feature tensor
    adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    features = np.zeros((NUM_ATOMS, ATOM_DIM), "float32")

    # loop over each atom in molecule
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        features[i] = np.eye(ATOM_DIM)[atom_type]
        # loop over one-hop neighbors
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, [i, j], [j, i]] = 1

    # Where no bond, add 1 to last channel (indicating "non-bond")
    # Notice: channels-first
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1

    # Where no atom, add 1 to last column (indicating "non-atom")
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

    return adjacency, features

def graph_to_molecule(graph):
    '''
    Reference: https://keras.io/examples/generative/wgan-graphs/
    '''
    # Unpack graph
    adjacency, features = graph

    # RWMol is a molecule object intended to be edited
    molecule = Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where(
        (np.argmax(features, axis=1) != ATOM_DIM - 1)
        & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
    )[0]
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule
    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        _ = molecule.AddAtom(atom)

    # Add bonds between atoms in molecule; based on the upper triangles
    # of the [symmetric] adjacency tensor
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atom_i), int(atom_j), bond_type)

    # Sanitize the molecule; for more information on sanitization, see
    # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule
```
- These two functions, **smiles_to_graph** and **graph_to_molecule**, are designed to convert between SMILES strings and molecular graph representations suitable for deep learning models. Here's a breakdown of each function:

1. smiles_to_graph(smiles):
- Input: A SMILES string representing a molecule.
- Conversion Steps:
- Uses Chem.MolFromSmiles to create an RDKit molecule object from the SMILES string.
    - Initializes two tensors:
    - adjacency: A 3D tensor of size (BOND_DIM, NUM_ATOMS, NUM_ATOMS). This tensor will represent the connections between atoms and the bond types.
    - features: A 2D tensor of size (NUM_ATOMS, ATOM_DIM). This tensor will represent features associated with each atom.
- Iterates over each atom in the molecule:
    - Gets the atom's index (i), atomic symbol, and corresponding feature vector based on the atom mapping dictionary.
- Iterates over the atom's neighbors:
    - Gets the neighbor's index (j) and the bond type between them.
    - Uses the bond mapping dictionary to get the corresponding bond type index.
    - Sets the adjacency tensor element at the specific bond type channel, atom indices (i, j), and their reversed order (j, i) to 1, indicating a bond connection.
- Handles "no bond" and "no atom" cases:
    - In the adjacency tensor's last channel (representing "non-bond"), sets elements to 1 where there are no bonds between atoms.
    - In the features tensor's last column (representing "non-atom"), sets elements to 1 for rows where the atom type is "non-atom" (based on feature vector sum).
Returns the adjacency and features tensors, representing the molecular graph.
2. graph_to_molecule(graph):
- Input: A tuple containing the adjacency and features tensors from the smiles_to_graph function.
- Conversion Steps:
    1. Unpacks the graph (adjacency and features tensors).
    2. Identifies and removes "no atom" and atom entries with no bonds based on the features and adjacency tensors.
    3. Creates an RDKit RWMol object, allowing for molecule editing.
    4. Iterates over the remaining features (excluding "non-atom" entries) and creates corresponding atoms in the molecule based on the atom type mapping.
    5. Iterates over the upper triangle (excluding diagonal and "non-bond" channel) of the adjacency tensor to identify connected atom pairs and their corresponding bond type index.
    6. For each connected atom pair:
        - Skips entries with the same atom index or "non-bond" type.
        - Adds a bond between the atoms in the molecule based on the retrieved bond type.
    7. Sanitizes the molecule using Chem.SanitizeMol. This ensures the generated molecule has a valid chemical structure.
    8. If sanitization fails, returns None.
    - Returns the sanitized RDKit molecule object.
 
- Overall, these functions provide a way to convert between SMILES strings (textual representation) and a graph-based representation suitable for deep learning models. The graph representation uses tensors to encode atom features and connections with bond types.
