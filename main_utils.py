    #取10个channel输出看一看其channel
    for i in range(0,10) :
        # feat = feat_extract(k,[w_featmap,h_featmap],i)
        feat = k[:,:,i].unsqueeze(2)
        feats.append(feat)


