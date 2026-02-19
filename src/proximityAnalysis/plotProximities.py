####################### gammaH2AX - CD8 proximity ##################################

print('Analyzing gammaH2AX and CD8 proximity...')
start_time = time.time()
neighbors_gh2ax = {}
neighbors_cd8 = {}
neighbors_random_gh2ax = {}
neighbors_random_cd8 = {}
for idx,c in enumerate(CLASSES):
    print(f'CLASS: {c}')
    tree = ET.parse(XML_FILES_PATH[idx])
    root = tree.getroot()
    neighbors_gh2ax[c] = []
    neighbors_random_gh2ax[c] = []
    for stack in root.iter('Stack'):
        col,row = int(stack.get('COL')), int(stack.get('ROW'))
        print(f'col: {col}, row: {row}')

        cd8_segment_file = 'tile__H%03d_V%03d_cd8.tif' % (col,row)
        gh2ax_segment_file = 'tile__H%03d_V%03d_gh2ax.tif' % (col,row)
        nuclei_segment_file = 'tile__H%03d_V%03d_CELLPOSE-LABELS_cp_masks.tif' % (col,row)

        print(f"READING: {cd8_segment_file=}, {gh2ax_segment_file=}, {nuclei_segment_file=}")

        cd8_segmentation = tifffile.imread(os.path.join(SEGMENTATION_PATH[idx], cd8_segment_file))
        gh2ax_segmentation = tifffile.imread(os.path.join(SEGMENTATION_PATH[idx], gh2ax_segment_file))
        nuclei_segmentation = tifffile.imread(os.path.join(SEGMENTATION_PATH[idx], nuclei_segment_file))

        labels = np.unique(gh2ax_segmentation)
        labels = labels[labels != 0]  # Exclude background label

        # Parallelize the label processing
        n_jobs = os.cpu_count() // 2  # Adjust based on your system
        print(f'os.cpu_count: {os.cpu_count()}, n_jobs: {n_jobs}')
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_label)(lbl, gh2ax_segmentation, cd8_segmentation, radii) for lbl in tqdm(labels))

        # Filter out None results and append to neighbors_gh2ax
        start_time_inner = time.time()
        for result in results:
            if result is not None:
                lbl, neighbors = result
                neighbors_gh2ax[c].append(neighbors)

        elapsed_time = time.time() - start_time_inner
        print(f"Time taken for finding neighbors of {c}, row: {row}, col: {col}: {elapsed_time} seconds")

        # random_labels = np.random.permutation(nuclei_segmentation.max())[:100] + 1
        # for lbl in random_labels:
        # 	n = get_neighbors(lbl, nuclei_segmentation, cd8_segmentation, radius=radii)
        # 	neighbors_random_gh2ax[c].append(n)
        # 	n = get_neighbors(lbl, nuclei_segmentation, gh2ax_segmentation, radius=radii)
        # 	neighbors_random_cd8[c].append(n)

for i in CLASSES:
    neighbors_gh2ax[i] = np.array(neighbors_gh2ax[i])
    np.savetxt('gh2ax_to_proximity_%s.csv'%(i), neighbors_gh2ax[i], delimiter=',')

t,p = stats.ttest_ind(neighbors_gh2ax[CLASSES[0]],neighbors_gh2ax[CLASSES[1]],equal_var=False)
print('gammaH2AX (control vs treated) u-test p-values:', p)

for idx,v in enumerate(radii):
    plt.boxplot([neighbors_gh2ax[i][:,idx].flatten()-1 for i in CLASSES], showfliers=False, meanline=True)
    # plt.ylim(0,0.15)
    plt.xticks([1,2], CLASSES)
    plt.ylabel('# of neighbor CD8+ cells')
    plt.title('gammaH2AX proximity to CD8 (%dpx)' % (v))
    plt.savefig('gammaH2AX proximity to CD8 (%dpx).png' % (v))
    plt.clf()

elapsed_time = time.time() - start_time
print(f"Time taken for the rest of the gammaH2AX - CD8 proximity analysis: {elapsed_time} seconds")
print('')

elapsed_time = time.time() - start_time0
print(f'Time taken for the entire analysis.py run: {elapsed_time} seconds')