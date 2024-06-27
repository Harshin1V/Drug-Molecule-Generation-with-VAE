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
