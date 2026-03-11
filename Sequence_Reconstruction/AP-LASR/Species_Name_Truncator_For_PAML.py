import os
import pandas as pd

# Define the input and output file paths
input_file = r"C:\Users\anton\Downloads\Final_Sequences.fasta"
output_file = r"C:\Users\anton\Downloads\Final_Sequences_Short.fasta"
excel_output_file = r"C:\Users\anton\Downloads\Species_Names.xlsx"

def shorten_species_name(header):
    """
    Shorten the species name in the header to a maximum of 10 characters (excluding the '>').
    """
    return header[1:11]  # Remove '>' and take the first 10 characters

def process_fasta(input_file, output_file, excel_output_file):
    """
    Process the input FASTA file, write the output to a new file with shortened species names,
    and generate an Excel sheet with the original and truncated names.
    """
    original_names = []
    truncated_names = []

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith('>'):
                header = line.strip()
                shortened_header = shorten_species_name(header)
                outfile.write(f">{shortened_header}\n")
                
                # Store the original and truncated names
                original_names.append(header)
                truncated_names.append(f">{shortened_header}")
            else:
                outfile.write(line)

    # Create a DataFrame and save it as an Excel file
    df = pd.DataFrame({
        "Original Species Names": original_names,
        "Truncated Species Names": truncated_names
    })
    df.to_excel(excel_output_file, index=False)

# Run the script
process_fasta(input_file, output_file, excel_output_file)
print("Processing complete. The shortened sequences are saved in:", output_file)
print("Excel sheet with species names saved as:", excel_output_file)