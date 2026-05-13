from utils import *
from train import *
from visualization import *
import json 
from Modules import MultiFeatMix


def get_seeds(dataset):
    if dataset == 'Lee':
        testseeds = [735715, 468261, 884807, 408141, 335839, 598764, 353617, 664118, 104908, 822596,
                    129898, 682656, 318109, 879136, 730637, 557627, 753260, 315788, 186666, 243967,
                    960948, 972593, 185272, 17889, 221168, 380684, 653430, 132226, 890591, 820294,
                    40803, 162059, 503280, 944383, 468642, 557513, 357992, 79616, 563898, 483552,
                    522696, 831588, 128350, 600368, 248121, 777091, 645281, 668028, 755110, 883476]
        batchsize=256

    elif dataset == '4DN':
        testseeds = [884807, 408141, 640905, 353617, 129898, 682656, 318109, 879136,
                    730637, 557627, 753260, 553225, 972593, 185272, 221168, 380684,
                    820294, 209523, 162059, 503280, 468642, 557513, 527263, 522696, 32041,
                    668028, 111881, 503384, 677611, 631852, 221929, 304937, 914569, 282637,
                    664421, 751662, 124891, 954318, 917249, 5748, 2624, 5008, 820, 6427, 5334,
                    2486, 5885, 3662, 2975, 576]
        batchsize=256

    elif dataset == 'Collombet':
        testseeds = [401523, 735715, 468261, 884807, 408141, 335839, 640905, 598764,
                    978791, 664118, 822596, 129898, 682656, 318109, 879136, 557627, 753260,
                    553225, 186666, 243967, 960948, 972593, 668572, 17889, 221168, 73610,
                    653430, 132226, 739264, 890591, 820294, 40803, 209523, 162059, 503280,
                    944383, 468642, 357992, 527263, 68755, 79616, 563898, 483552, 522696,
                    831588, 128350, 600368, 248121, 777091, 645281]
        # testseeds = [40803, 248121, 282637, 221929, 712404, 1459]
        batchsize=64
        
    elif dataset == 'Nagano':
        testseeds = [401523, 735715, 468261, 884807, 408141, 335839, 640905, 598764, 978791, 353617,
                    664118, 104908, 822596, 129898, 682656, 318109, 879136, 730637, 557627, 753260,
                    553225, 315788, 186666, 243967, 960948, 972593, 185272, 668572, 17889, 221168,
                    380684, 73610, 653430, 132226, 739264, 890591, 820294, 40803, 209523, 162059,
                    503280, 944383, 468642, 557513, 357992, 527263, 68755, 79616, 563898, 483552]
        batchsize=64

    elif dataset == 'Ramani':
        testseeds = [401523, 735715, 468261, 335839, 104908, 129898, 879136, 186666, 960948, 972593,
                    185272, 17889, 132226, 739264, 40803, 209523, 162059, 527263, 483552, 522696,
                    32041, 600368, 248121, 645281, 755110, 609102, 213095, 221929, 282637, 603207,
                    58851, 954318, 308263, 425632, 712404, 4678, 2763, 1459, 2975, 5105,
                    3778, 5900, 602, 5294, 4937, 1496, 6104, 6659, 1186, 4178]
        batchsize=64
    
    return testseeds, batchsize
    

if __name__ == '__main__':
    dataset = sys.argv[1]
    print(f'dataset:{dataset}')
    
    load_model = False

    config_path = os.path.join('../datasets',dataset,'config.json')
    with open(config_path,'r') as f:
        config = json.load(f)

    trainer = Train(config, no_tqdm=True)
    trainer.prep_data()
    trainer.prep_model()
    labels = trainer.labels
    cell_feats = trainer.cell_init.clone().cpu().detach().numpy()
    
    import xlsxwriter
    os.makedirs('../class_res',exist_ok=True)
    file = f'../class_res/{dataset}.xlsx'
    workbook = xlsxwriter.Workbook(file)
    worksheet1 = workbook.add_worksheet(f'{dataset}')
    worksheet1.write(0, 0, 'test_seed')
    worksheet1.write(0, 1, 'acc')
    worksheet1.write(0, 2, 'micro_F1')
    worksheet1.write(0, 3, 'macro_F1')
    worksheet1.write(0, 4, 'micro_Precision')
    worksheet1.write(0, 5, 'macro_Precision')
    worksheet1.write(0, 6, 'bacc')
    worksheet1.write(0, 7, 'mcc')
    worksheet1.write(0, 8, 'micro_Recall')
    worksheet1.write(0, 9, 'macro_Recall')
    row = 0

    results_collector = {}
    if load_model:
        trainer.load_model()
    else:
        print("not load model!")
        encoder_dims = [cell_feats.shape[-1], trainer.dimensions]
        feats_num = cell_feats.shape[1]

    results_list = []
    testseeds, batchsize = get_seeds(dataset)
    
    for i,testseed in enumerate(testseeds):
        set_random_seed(testseed)
        print(f'[{i+1}/{len(testseeds)}] testseed: {testseed}')
        if load_model:
            encoder = copy.deepcopy(trainer.head_model.cell_encoder)
        else:
            encoder = MultiFeatMix(encoder_dims[0], feats_num, encoder_dims[-1])
        
        class_res = classify(labels=labels, cell_init=cell_feats, 
                            cell_encoder=encoder, no_tqdm=True, batch_size=batchsize, seed=testseed)

        
        std_print(class_res.keys(),class_res.values(),style='raw')
        results_list.append(class_res)

        for metric_name, value in class_res.items():
            results_collector.setdefault(metric_name, []).append(value)
            
        row += 1
        worksheet1.write(row, 0, testseed)
        worksheet1.write(row, 1, class_res['acc'])
        worksheet1.write(row, 2, class_res['micro_F1'])
        worksheet1.write(row, 3, class_res['macro_F1'])
        worksheet1.write(row, 4, class_res['micro_Precision'])
        worksheet1.write(row, 5, class_res['macro_Precision'])
        worksheet1.write(row, 6, class_res['bacc'])
        worksheet1.write(row, 7, class_res['mcc'])
        worksheet1.write(row, 8, class_res['micro_Recall'])
        worksheet1.write(row, 9, class_res['macro_Recall'])

        if (i+1) % 5 == 0:
            for metric_name, scores_list in results_collector.items():
                scores_array = np.array(scores_list)
                mean_score = np.mean(scores_array)
                std_score = np.std(scores_array)
                
                print(f"{metric_name:<10}: {mean_score:.4f} ± {std_score:.4f}")
                
    workbook.close()

    