import os
import zipfile
import pandas as pd
import numpy as np
import io

def prepare_dataset(zip_path="Stanford_Online_Products.zip", output_dir="data", n_samples_per_category=150):
    print("=" * 60)
    print("Step 1: Extracting and Preparing Stanford Online Products Dataset")
    print("=" * 60)
    
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    category_map = {
        1: "Bicycle",
        2: "Cabinet",
        3: "Chair",
        4: "Coffee Maker",
        5: "Fan",
        6: "Kettle",
        7: "Lamp",
        8: "Mug",
        9: "Sofa",
        10: "Stapler",
        11: "Table",
        12: "Toaster"
    }

    price_ranges = {
        "Bicycle": (199.99, 899.99),
        "Cabinet": (149.99, 599.99),
        "Chair": (49.99, 299.99),
        "Coffee Maker": (29.99, 179.99),
        "Fan": (24.99, 89.99),
        "Kettle": (19.99, 69.99),
        "Lamp": (29.99, 129.99),
        "Mug": (8.99, 24.99),
        "Sofa": (299.99, 1299.99),
        "Stapler": (7.99, 29.99),
        "Table": (89.99, 449.99),
        "Toaster": (19.99, 79.99)
    }

    print(f"Reading metadata from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        info_bytes = z.read("Stanford_Online_Products/Ebay_info.txt")
        df = pd.read_csv(io.BytesIO(info_bytes), sep=' ')
        
        # Map super_class_id to category
        df['category'] = df['super_class_id'].map(category_map)
        
        # Balance sample across categories
        sampled_dfs = []
        for cat_id, cat_name in category_map.items():
            cat_df = df[df['super_class_id'] == cat_id]
            # Sample deterministically with fixed random state for reproducibility
            sample = cat_df.sample(n=min(n_samples_per_category, len(cat_df)), random_state=42)
            sampled_dfs.append(sample)
            
        final_df = pd.concat(sampled_dfs).reset_index(drop=True)
        print(f"Sampled {len(final_df)} images across {len(category_map)} categories ({n_samples_per_category} per category).")
        
        print("Extracting selected images to data/images/...")
        extracted_paths = []
        np.random.seed(42)
        
        for idx, row in final_df.iterrows():
            zip_internal_path = "Stanford_Online_Products/" + row['path']
            filename = f"{row['category'].lower().replace(' ', '_')}_{row['image_id']}.jpg"
            dest_path = os.path.join(images_dir, filename)
            
            # Extract only if not already extracted
            if not os.path.exists(dest_path):
                img_data = z.read(zip_internal_path)
                with open(dest_path, "wb") as f_out:
                    f_out.write(img_data)
            
            # Save relative path for cross-platform portability
            extracted_paths.append(os.path.join("data", "images", filename).replace("\\", "/"))
            
            if (idx + 1) % 300 == 0 or (idx + 1) == len(final_df):
                print(f"Extracted {idx + 1}/{len(final_df)} images...")
                
        final_df['local_image_path'] = extracted_paths
        
        # Generate Amazon-style product attributes for realistic e-commerce experience
        titles = []
        prices = []
        ratings = []
        review_counts = []
        adjectives = ["Classic", "Modern", "Ergonomic", "Minimalist", "Premium", "Vintage", "Compact", "Deluxe", "Commercial Grade", "Pro"]
        
        for idx, row in final_df.iterrows():
            cat = row['category']
            adj = adjectives[row['class_id'] % len(adjectives)]
            titles.append(f"{adj} {cat} (SKU-{row['class_id']:05d})")
            
            low, high = price_ranges.get(cat, (20.0, 100.0))
            price = round(np.random.uniform(low, high), 2)
            prices.append(price)
            
            rating = round(np.random.uniform(3.8, 4.9), 1)
            ratings.append(rating)
            
            reviews = int(np.random.randint(15, 850))
            review_counts.append(reviews)
            
        final_df['title'] = titles
        final_df['price_usd'] = prices
        final_df['rating'] = ratings
        final_df['review_count'] = review_counts
        
        # Select and save clean metadata
        clean_metadata = final_df[[
            'image_id', 'class_id', 'super_class_id', 'category', 
            'title', 'price_usd', 'rating', 'review_count', 'local_image_path'
        ]]
        
        meta_csv_path = os.path.join(output_dir, "metadata.csv")
        clean_metadata.to_csv(meta_csv_path, index=False)
        print(f"Clean metadata saved to: {meta_csv_path}")
        print("\nDataset summary by category:")
        print(clean_metadata['category'].value_counts())
        print("\nFirst 3 records:")
        print(clean_metadata[['category', 'title', 'price_usd', 'rating', 'local_image_path']].head(3))
        print("Data extraction and preparation successfully completed!")

if __name__ == "__main__":
    prepare_dataset()
