class modification_reader:
    """
    Reads manual annotation file.
    Supports:
      - Name: id1 id2 ...
      - Name: id*start-end (time-bounded assignment)
      - Name: id1, id2, id3*start-end (multi-id time-bounded assignment)
      - Name: id1*range, id2*range (comma-separated assignments)
      - SWAP: frame id1 id2
    """
    def __init__(self, text_file_path, rewrite=False):
        self.text_file_path = text_file_path
        self.swaps = {}
        self.rewrite = rewrite
        self.id_to_name_intervals = {}   # {id: [(name, start, end), ...]}
        self.id_to_name_global = {}      # {id: name} for full-video assignments
        self.read()

    def _parse_token(self, token):
        # token examples: "10", "10,11", "10*650-1000", "10,11*650-"
        
        # Check if this token has a time range (marked by *)
        if "*" not in token:
            # No range, just IDs (potentially comma separated like "1,2,3")
            ids = token.split(',')
            return ids, None, None
            
        id_part, rng = token.split("*", 1)
        
        # Parse the range part
        if "-" in rng:
            start_s, end_s = rng.split("-", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else float("inf")
        else:
            start = int(rng)
            end = float("inf")
            
        # Parse the ID part (split by commas)
        ids = id_part.split(',')
        
        return ids, start, end

    def read(self):
        parsed_content = []
        unknown_id_index = 0
        with open(self.text_file_path, 'r') as text_file:
            for line in text_file.read().split("\n"):
                if len(line) < 1:
                    continue
                if ":" in line:  # name present
                    name, rhs = line.split(": ", 1)
                    
                    # Handle SWAP separately first
                    if name.upper() == "SWAP":
                        frame_count, swap_id_1, swap_id_2 = rhs.split(" ")
                        for a, b in [(swap_id_1, swap_id_2), (swap_id_2, swap_id_1)]:
                            self.swaps.setdefault(a, []).append((int(frame_count), b))
                        continue
                    
                    # Normalize spaces around commas: "1, 2, 3" -> "1,2,3"
                    rhs = re.sub(r'\s*,\s*', ',', rhs)
                    token_pattern = r'[^,*\s]+(?:,[^,*\s]+)*(?:\*[\d\-]+)?'
                    tokens = re.findall(token_pattern, rhs)
                    
                    for tok in tokens:
                        if not tok: continue
                        
                        id_list, start, end = self._parse_token(tok)
                        
                        for cid in id_list:
                            # Handle Global Assignment (no *)
                            if start is None:
                                self.id_to_name_global[cid] = name
                            # Handle Interval Assignment (has *)
                            else:
                                self.id_to_name_intervals.setdefault(cid, []).append((name, start, end))
                                
                    parsed_content.append([name, tokens])
                else:
                    name = f"UNK_{unknown_id_index}"
                    parsed_content.append([name, line.split(" ")])
                    unknown_id_index += 1
        
        if self.rewrite:
            content = ""
            for name, numbers in parsed_content:
                content += f"{name}: {' '.join(str(n) for n in numbers)}\n"
            with open(self.text_file_path, "w") as f:
                f.write(content)
        self.data = parsed_content