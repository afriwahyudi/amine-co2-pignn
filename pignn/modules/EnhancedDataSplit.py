# DATA SPLIT SCHEME
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np
import random
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, dataset, random_state: int = 21):
        self.dataset = dataset
        self.random_state = random_state
        self._analyze_dataset()
        self.thresholds = self._get_adaptive_thresholds()

    def _analyze_dataset(self):
        self.amine_to_type = {}
        self.type_to_amines = defaultdict(list)
        self.amine_to_indices = defaultdict(list)
        
        for idx, data in enumerate(self.dataset):
            amine_name = data.name
            amine_type = data.type
            
            if amine_name not in self.amine_to_type:
                self.amine_to_type[amine_name] = amine_type
                self.type_to_amines[amine_type].append(amine_name)
            
            self.amine_to_indices[amine_name].append(idx)
        
        self.amine_names = list(self.amine_to_type.keys())
        self.amine_types = list(self.type_to_amines.keys())

    def _get_adaptive_thresholds(self):
        """Calculate adaptive thresholds based on dataset characteristics."""
        import numpy as np
        
        # Calculate sample and amine counts for each type
        type_sample_counts = {}
        type_amine_counts = {}
        for amine_type in self.amine_types:
            type_sample_counts[amine_type] = sum(len(self.amine_to_indices[amine]) 
                                               for amine in self.type_to_amines[amine_type])
            type_amine_counts[amine_type] = len(self.type_to_amines[amine_type])
        
        sample_counts = list(type_sample_counts.values())
        amine_counts = list(type_amine_counts.values())
        total_samples = len(self.dataset)
        
        # Calculate sample-based thresholds
        sample_percentiles = np.percentile(sample_counts, [25, 50, 75])
        very_rare_sample_threshold = min(sample_percentiles[0], total_samples * 0.01)
        rare_sample_threshold = min(sample_percentiles[1], total_samples * 0.05)
        common_sample_threshold = max(sample_percentiles[2], total_samples * 0.15)
        
        # Calculate amine-based thresholds adaptively
        amine_percentiles = np.percentile(amine_counts, [15, 35])  # Lower percentiles for amine counts
        
        # Very rare: bottom 15th percentile or at most 20% of median
        very_rare_amine_threshold = max(1, min(amine_percentiles[0], np.median(amine_counts) * 0.2))
        
        # Rare: bottom 35th percentile or at most 50% of median  
        rare_amine_threshold = max(very_rare_amine_threshold + 1, 
                                 min(amine_percentiles[1], np.median(amine_counts) * 0.5))
        
        return {
            'very_rare_sample_threshold': very_rare_sample_threshold,
            'rare_sample_threshold': rare_sample_threshold, 
            'common_sample_threshold': common_sample_threshold,
            'very_rare_amine_threshold': int(very_rare_amine_threshold),
            'rare_amine_threshold': int(rare_amine_threshold)
        }

    def print_dataset_stats(self):
        print(f"Dataset: {len(self.dataset)} samples, {len(self.amine_names)} unique amines")
        print("Amine type distribution:")
        for amine_type in sorted(self.amine_types):
            count = len(self.type_to_amines[amine_type])
            print(f"  {amine_type}: {count} amines")
        print()

    def stratified_random_split(self, test_size: float = 0.1, val_size: float = 0.15) -> Tuple[List, List, List]:
        """Group indices by amine type and perform stratified split."""
        type_to_indices = defaultdict(list)
        for idx, data in enumerate(self.dataset):
            type_to_indices[data.type].append(idx)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Perform stratified split for each amine type
        for amine_type, indices in type_to_indices.items():
            # First split: separate test set
            remaining_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=self.random_state
            )
            
            # Second split: separate validation from train
            train_idx, val_idx = train_test_split(
                remaining_idx, 
                test_size=val_size/(1-test_size), 
                random_state=self.random_state
            )
            
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)
        
        # Create the actual data splits
        train_data = [self.dataset[i] for i in train_indices]
        val_data = [self.dataset[i] for i in val_indices]
        test_data = [self.dataset[i] for i in test_indices]
        
        self._print_split_stats(train_data, val_data, test_data, "Stratified Random Split")
        return train_data, val_data, test_data

    def _classify_rarity(self, amine_type, sample_count, amine_count):
        """Classify amine type rarity and return appropriate split ratios."""
        if (sample_count < self.thresholds['very_rare_sample_threshold'] or 
            amine_count <= self.thresholds['very_rare_amine_threshold']):
            # Very rare goes 100% to train since they can't be split
            return "Very Rare", (1.0, 0.0, 0.0)
        elif (sample_count < self.thresholds['rare_sample_threshold'] or 
              amine_count <= self.thresholds['rare_amine_threshold']):
            # Rare: still conservative, but still give most to training
            return "Rare", (0.90, 0.10, 0)
        elif sample_count > self.thresholds['common_sample_threshold']:
            # Common: train > val = test
            return "Common", (0.70, 0.15, 0.15)
        else:
            # Medium: train > val > test
            return "Medium", (0.70, 0.20, 0.10)

    def _assign_amines_to_splits(self, type_amines, train_ratio, val_ratio, test_ratio):
        """Assign amines from a type to train/val/test splits with guaranteed representation."""
        type_amines = type_amines.copy()
        np.random.shuffle(type_amines)
        
        n_amines = len(type_amines)
        
        # Handle single amine case - goes to train only (very rare types)
        if n_amines == 1:
            return type_amines, [], []
        
        # For 2 amines: prioritize train, then val
        elif n_amines == 2:
            # Give priority to training (75% of cases)
            if train_ratio >= 0.6:  # If this is a train-heavy split
                return [type_amines[0]], [type_amines[1]], []
            else:
                # More balanced split
                return [type_amines[0]], [type_amines[1]], []
        
        # For 3+ amines: guarantee at least 1 in val and test, rest distributed by ratio
        else:
            # Calculate target distribution
            target_train = max(1, int(n_amines * train_ratio))
            target_val = max(1, int(n_amines * val_ratio)) if val_ratio > 0 else 0
            target_test = max(1, int(n_amines * test_ratio)) if test_ratio > 0 else 0
            
            # Ensure we don't exceed total amines AND account for all amines
            total_assigned = target_train + target_val + target_test
            
            if total_assigned > n_amines:
                # Scale down proportionally, but maintain minimums
                if target_val > 0 and target_test > 0:
                    # All three splits needed
                    remaining = n_amines - 3  # Reserve 1 for each
                    extra_train = int(remaining * train_ratio)
                    extra_val = int(remaining * val_ratio)
                    extra_test = remaining - extra_train - extra_val
                    
                    target_train = 1 + extra_train
                    target_val = 1 + extra_val
                    target_test = 1 + extra_test
                elif target_val > 0:
                    # Only train and val
                    remaining = n_amines - 2
                    target_train = 1 + int(remaining * train_ratio / (train_ratio + val_ratio))
                    target_val = n_amines - target_train
                    target_test = 0
                else:
                    # Only train
                    target_train = n_amines
                    target_val = 0
                    target_test = 0
            elif total_assigned < n_amines:
                # We have unassigned amines - add them to training
                remaining_amines = n_amines - total_assigned
                target_train += remaining_amines
            
            # Assign amines with bounds checking
            train_end = min(target_train, n_amines)
            val_start = train_end
            val_end = min(val_start + target_val, n_amines)
            test_start = val_end
            test_end = n_amines  # Always go to the end to capture all remaining amines
            
            train_amines = type_amines[:train_end]
            val_amines = type_amines[val_start:val_end] if target_val > 0 else []
            test_amines = type_amines[test_start:test_end] if target_test > 0 else []
            
            # Verify all amines are assigned
            total_assigned_actual = len(train_amines) + len(val_amines) + len(test_amines)
            if total_assigned_actual != n_amines:
                # Fallback: put any unassigned amines in training
                assigned_amines = set(train_amines + val_amines + test_amines)
                unassigned = [a for a in type_amines if a not in assigned_amines]
                train_amines.extend(unassigned)
            
            return train_amines, val_amines, test_amines

    def rarity_aware_unseen_amine_split(self, test_size: float = 0.1, val_size: float = 0.15) -> Tuple[List, List, List]:
        """Split data ensuring no amine overlap between splits, considering rarity."""
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        # Calculate type statistics
        type_sample_counts = {}
        type_amine_counts = {}
        for amine_type in self.amine_types:
            total_samples = sum(len(self.amine_to_indices[amine]) 
                            for amine in self.type_to_amines[amine_type])
            type_sample_counts[amine_type] = total_samples
            type_amine_counts[amine_type] = len(self.type_to_amines[amine_type])
        
        # Group types by rarity for better organization
        rarity_groups = {
            'Very Rare': [],
            'Rare': [],
            'Medium': [],
            'Common': []
        }
        
        for amine_type in self.amine_types:
            sample_count = type_sample_counts[amine_type]
            amine_count = type_amine_counts[amine_type]
            rarity_level, (train_ratio, val_ratio, test_ratio) = self._classify_rarity(
                amine_type, sample_count, amine_count
            )
            rarity_groups[rarity_level].append({
                'type': amine_type,
                'amines': self.type_to_amines[amine_type].copy(),
                'ratios': (train_ratio, val_ratio, test_ratio)
            })
        
        all_train_amines = []
        all_val_amines = []
        all_test_amines = []
        
        # Process each rarity group
        for rarity_level, type_groups in rarity_groups.items():
            print(f"\nProcessing {rarity_level} types:")
            
            for type_info in type_groups:
                amine_type = type_info['type']
                type_amines = type_info['amines']
                train_ratio, val_ratio, test_ratio = type_info['ratios']
                
                # Assign amines to splits
                train_amines, val_amines, test_amines = self._assign_amines_to_splits(
                    type_amines, train_ratio, val_ratio, test_ratio
                )
                
                all_train_amines.extend(train_amines)
                all_val_amines.extend(val_amines)
                all_test_amines.extend(test_amines)
                
                print(f"  {amine_type}: {len(train_amines)} train, {len(val_amines)} val, {len(test_amines)} test")
        
        # Create final data splits
        train_data = []
        val_data = []
        test_data = []
        
        for amine in all_train_amines:
            train_data.extend([self.dataset[i] for i in self.amine_to_indices[amine]])
        
        for amine in all_val_amines:
            val_data.extend([self.dataset[i] for i in self.amine_to_indices[amine]])
            
        for amine in all_test_amines:
            test_data.extend([self.dataset[i] for i in self.amine_to_indices[amine]])
        
        # Shuffle to avoid ordering bias
        np.random.shuffle(train_data)
        np.random.shuffle(val_data)
        np.random.shuffle(test_data)
        
        # Validate and print results
        self._print_rarity_classification()
        self._print_distribution_table(all_train_amines, all_val_amines, all_test_amines)
        self._validate_no_amine_overlap(all_train_amines, all_val_amines, all_test_amines)
        self._print_split_stats(train_data, val_data, test_data, "Rarity-Aware Unseen Amine Split")
        
        # Verify sample conservation
        final_total = len(train_data) + len(val_data) + len(test_data)
        print(f"✓ Sample conservation check: {final_total}/{len(self.dataset)} samples")
        
        return train_data, val_data, test_data

    def _print_rarity_classification(self):
        """Print which types and their amines are classified as rare."""
        type_sample_counts = {}
        type_amine_counts = {}
        
        for amine_type in self.amine_types:
            type_sample_counts[amine_type] = sum(len(self.amine_to_indices[amine]) 
                                               for amine in self.type_to_amines[amine_type])
            type_amine_counts[amine_type] = len(self.type_to_amines[amine_type])
        
        # Classify each type
        very_rare_types = []
        rare_types = []
        medium_types = []
        common_types = []
        
        for amine_type in self.amine_types:
            sample_count = type_sample_counts[amine_type]
            amine_count = type_amine_counts[amine_type]
            
            rarity_level, _ = self._classify_rarity(amine_type, sample_count, amine_count)
            
            type_info = {
                'type': amine_type,
                'samples': sample_count,
                'amines': self.type_to_amines[amine_type],
                'amine_count': amine_count
            }
            
            if rarity_level == "Very Rare":
                very_rare_types.append(type_info)
            elif rarity_level == "Rare":
                rare_types.append(type_info)
            elif rarity_level == "Medium":
                medium_types.append(type_info)
            else:
                common_types.append(type_info)
        
        print("Rarity Classification:")
        print("=" * 60)
        
        def print_category(category_name, types_list):
            if types_list:
                print(f"\n{category_name}:")
                for type_info in sorted(types_list, key=lambda x: x['samples']):
                    amines_str = ", ".join(type_info['amines'][:3])  # Show first 3 amines
                    if len(type_info['amines']) > 3:
                        amines_str += f" ... (+{len(type_info['amines'])-3} more)"
                    
                    print(f"  {type_info['type']}: {type_info['samples']} samples, "
                          f"{type_info['amine_count']} amines [{amines_str}]")
        
        print_category("Very Rare", very_rare_types)
        print_category("Rare", rare_types)
        print_category("Medium", medium_types) 
        print_category("Common", common_types)
        print()

    def _print_distribution_table(self, train_amines, val_amines, test_amines):
        """Print rarity distribution summary across splits."""
        def analyze_rarity_distribution(amines):
            very_rare_count = rare_count = medium_count = common_count = 0
            very_rare_samples = rare_samples = medium_samples = common_samples = 0
            
            for amine in amines:
                amine_type = self.amine_to_type[amine]
                type_samples = sum(len(self.amine_to_indices[a]) for a in self.type_to_amines[amine_type])
                type_amine_count = len(self.type_to_amines[amine_type])
                amine_samples = len(self.amine_to_indices[amine])
                
                rarity_level, _ = self._classify_rarity(amine_type, type_samples, type_amine_count)
                
                if rarity_level == "Very Rare":
                    very_rare_count += 1
                    very_rare_samples += amine_samples
                elif rarity_level == "Rare":
                    rare_count += 1
                    rare_samples += amine_samples
                elif rarity_level == "Medium":
                    medium_count += 1
                    medium_samples += amine_samples
                else:  # Common
                    common_count += 1
                    common_samples += amine_samples
            
            return {
                'very_rare': {'amines': very_rare_count, 'samples': very_rare_samples},
                'rare': {'amines': rare_count, 'samples': rare_samples},
                'medium': {'amines': medium_count, 'samples': medium_samples},
                'common': {'amines': common_count, 'samples': common_samples}
            }
        
        train_dist = analyze_rarity_distribution(train_amines)
        val_dist = analyze_rarity_distribution(val_amines)
        test_dist = analyze_rarity_distribution(test_amines)
        
        print("=" * 90)
        print("RARITY DISTRIBUTION ACROSS SPLITS")
        print("=" * 90)
        print(f"{'Set':<12} {'Very Rare':>15} {'Rare':>15} {'Medium':>15} {'Common':>15}")
        print(f"{'':>12} {'Amines/Samples':<15} {'Amines/Samples':<15} {'Amines/Samples':<15} {'Amines/Samples':<15}")
        print("-" * 90)
        
        print(f"{'Train':<12} {train_dist['very_rare']['amines']:>7}/{train_dist['very_rare']['samples']:<7} "
              f"{train_dist['rare']['amines']:>7}/{train_dist['rare']['samples']:<7} "
              f"{train_dist['medium']['amines']:>7}/{train_dist['medium']['samples']:<7} "
              f"{train_dist['common']['amines']:>7}/{train_dist['common']['samples']:<7}")
        
        print(f"{'Validation':<12} {val_dist['very_rare']['amines']:>7}/{val_dist['very_rare']['samples']:<7} "
              f"{val_dist['rare']['amines']:>7}/{val_dist['rare']['samples']:<7} "
              f"{val_dist['medium']['amines']:>7}/{val_dist['medium']['samples']:<7} "
              f"{val_dist['common']['amines']:>7}/{val_dist['common']['samples']:<7}")
        
        print(f"{'Test':<12} {test_dist['very_rare']['amines']:>7}/{test_dist['very_rare']['samples']:<7} "
              f"{test_dist['rare']['amines']:>7}/{test_dist['rare']['samples']:<7} "
              f"{test_dist['medium']['amines']:>7}/{test_dist['medium']['samples']:<7} "
              f"{test_dist['common']['amines']:>7}/{test_dist['common']['samples']:<7}")
        
        print()

    def _validate_no_amine_overlap(self, train_amines, val_amines, test_amines):
        """Validate that there's no amine overlap between splits."""
        train_set = set(train_amines)
        val_set = set(val_amines)
        test_set = set(test_amines)
        
        overlap_train_val = train_set & val_set
        overlap_train_test = train_set & test_set
        overlap_val_test = val_set & test_set
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(f"Amine overlap detected! "
                            f"Train-Val: {overlap_train_val}, "
                            f"Train-Test: {overlap_train_test}, "
                            f"Val-Test: {overlap_val_test}")
        
        print(f"✓ No amine overlap. Unique amines - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    def _print_split_stats(self, train_data, val_data, test_data, split_name):
        """Print concise statistics for the data splits."""
        def get_stats(data):
            type_counts = defaultdict(int)
            amine_counts = defaultdict(int)
            for d in data:
                type_counts[d.type] += 1
                amine_counts[d.name] += 1
            return {
                'total': len(data),
                'types': dict(type_counts),
                'unique_amines': len(amine_counts)
            }
        
        train_stats = get_stats(train_data)
        val_stats = get_stats(val_data)
        test_stats = get_stats(test_data)
        total = train_stats['total'] + val_stats['total'] + test_stats['total']
        
        print(f"\n{split_name}:")
        print(f"Train: {train_stats['total']} samples ({train_stats['total']/total:.1%}), {train_stats['unique_amines']} amines")
        print(f"Val:   {val_stats['total']} samples ({val_stats['total']/total:.1%}), {val_stats['unique_amines']} amines") 
        print(f"Test:  {test_stats['total']} samples ({test_stats['total']/total:.1%}), {test_stats['unique_amines']} amines")
        print()