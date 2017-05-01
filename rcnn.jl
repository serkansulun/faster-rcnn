using Images, Knet, MAT, ImageMagick

function loss(w,im,label_gold,bbox_gold)
    lambda = 4
    (label_pred,bbox_pred) = forward_rpn(w,im)
    label_gold_vec = mat(label_gold)
    label_pred_vec = mat(label_pred)
    bbox_pred_vec = mat(bbox_pred)
    bbox_gold_vec = mat(bbox_gold)
    label_norm_vec = logp(label_pred_vec,1)
    # Log loss for labels
    loss_label=-sum(label_gold_vec .* label_norm_vec) / size(label_norm_vec,1)
    # Smooth L1 loss for bboxes
    diff = abs(bbox_gold_vec-bbox_pred_vec)
    #diff = convert(Array{Float32},diff)

    smoothL1 = zeros(size(diff,1),1)
    smoothL1 = convert(atype,smoothL1)

    smoothL1 += 0.5.*diff.*diff.*(diff.<1)
    smoothL1 += (diff .- 0.5).*(diff.>=1)

    # We only calculate bbox loss for positive labels so first create a labels
    # matrix as the same size as bbox, in order to multiply
    label_mult = zeros(Float32,size(bbox_pred))
    label_gold_f = convert(Array{Float32},label_gold)
    # repmat doesn't work in 3rd dimension so use a for loop
    for i=1:4
      label_mult[:,:,9*(i-1)+1:9*i,:] = label_gold_f
    end

    label_mult_vec = convert(atype,mat(label_mult))
    loss_bbox = -sum(label_mult_vec.*smoothL1) / size(smoothL1,1)

    lss = loss_label + lambda*loss_bbox

end

lossgradient = grad(loss)

function train(w, im, label_gold, bbox_gold,lr)#; iters=1800)

            g = lossgradient(w, im, label_gold, bbox_gold)

            for i in 1:length(w)

                #w[i] -= lr * g[i]
                axpy!(-lr, g[i], w[i])
								#update!(w[i],g[i],prms[i])
            end
            # if (iters -= 1) <= 0
            #     return w
            # end
    return w
end

function prepare_roi(anchors,gt_box,gt_label)
  # Calculate overlaps between anchor and gt boxes
  overlap = boxoverlap(anchors,gt_box)
  (anc_max_overlaps, anc_assignment) = mymax(overlap, 1);
  (gt_max_overlaps, gt_assignment) = mymax(overlap, 2);
  # find all anchors with max_gt_overlap
  matching=overlap.==gt_max_overlaps'
  matchall = falses(size(matching,1),1)
  for i = 1:size(matching,2)
   matchall = matchall | matching[:,i]
  end
  gt_best_match = find(matchall)  # foreground and background indices
  fg_inds = unique([find(anc_max_overlaps.>=0.7);gt_best_match])
  #bg_inds = find(anc_max_overlaps.<0.3)
  # drop background indices outside the image
  #bg_inds = intersect(bg_inds,find(!outside_image(anchors)))
  target_box = gt_box[anc_assignment[fg_inds], :]
  anc_box = anchors[fg_inds, :]
  regression_label = rcnn_bbox_transform(anc_box, target_box);
  bbox_targets = zeros(size(anchors, 1), 4);
  labels = zeros(size(anchors, 1))
  #labels[fg_inds] = gt_label[anc_assignment[fg_inds]]
  labels[fg_inds] = 1
  #labels[bg_inds] = -1
  bbox_targets[fg_inds, :] = regression_label
  label_rs = reshape(labels,32,32,9,1)
  bbox_rs = reshape(bbox_targets,32,32,9*4,1)

  return (convert(atype,label_rs),convert(atype,bbox_rs))
  #return (label_rs,bbox_rs)

end

function rcnn_bbox_transform(anc_boxes,gt_boxes)
  anc_widths = anc_boxes[:, 3] - anc_boxes[:, 1] + 1;
  anc_heights = anc_boxes[:, 4] - anc_boxes[:, 2] + 1;
  anc_ctr_x = anc_boxes[:, 1] .+ 0.5 .* (anc_widths - 1);
  anc_ctr_y = anc_boxes[:, 2] .+ 0.5 .* (anc_heights - 1);

  gt_widths = gt_boxes[:, 3] - gt_boxes[:, 1] + 1;
  gt_heights = gt_boxes[:, 4] - gt_boxes[:, 2] + 1;
  gt_ctr_x = gt_boxes[:, 1] .+ 0.5 .* (gt_widths - 1);
  gt_ctr_y = gt_boxes[:, 2] .+ 0.5 .* (gt_heights - 1);

  targets_dx = (gt_ctr_x - anc_ctr_x) ./ (anc_widths)
      targets_dy = (gt_ctr_y - anc_ctr_y) ./ (anc_heights)
      targets_dw = log(gt_widths ./ anc_widths)
      targets_dh = log(gt_heights ./ anc_heights)

    regression_label = hcat(targets_dx, targets_dy, targets_dw, targets_dh)

end

function outside_image(anchors)
  outside = (anchors.<=0) | (anchors.>500)
  out = outside[:,1] | outside[:,2] | outside[:,3] | outside[:,4]

end

function mymax(m,dim)
  n = size(m,dim)
  val = Array(Float64,n)
  ind = Array(Int64,n)
  for i = 1:n
    if dim==1
      (val[i],ind[i]) = findmax(m[i,:])
    else
      (val[i],ind[i]) = findmax(m[:,i])
    end
  end
  return (val,ind)
end

function boxoverlap(anchors,gt_box)
  iou = Array{Float64}(size(anchors,1),size(gt_box,1))
  for i = 1:size(gt_box,1)
    x1 = max(anchors[:,1],gt_box[i,1])
    y1 = max(anchors[:,2],gt_box[i,2])
    x2 = min(anchors[:,3],gt_box[i,3])
    y2 = min(anchors[:,4],gt_box[i,4])

    w = x2-x1+1
    h = y2-y1+1
    # Calculate intersection over union overlap
    inter = w.*h
    anc_area = (anchors[:,3]-anchors[:,1]+1) .* (anchors[:,4]-anchors[:,2]+1)
    gt_area = (gt_box[i,3]-gt_box[i,1]+1) .* (gt_box[i,4]-gt_box[i,2]+1)

    iou[:,i] =  inter./(anc_area.+gt_area-inter)
    # clear non overlapping samples
    iou[w.<0,i] =  0
    iou[h.<0,i] =  0

  end
  # clear anchors outside image

  iou[outside_image(anchors),:] = 0
  return iou
end

function readlabel(label_number)
  f = open(string(folder,"/dataset/train/annotations/",label_number))
  s = readlines(f)
  gt_label = Array{Int64}(length(s))
  gt_box = Array{Float32}(length(s),4)
  for i = 1:length(s)
    s_array = split(s[i])
    gt_label[i]=class[s_array[1]]
    gt_box[i,:] = [parse(Float32,s_array[2]),parse(Float32,s_array[3]),parse(Float32,s_array[4]),parse(Float32,s_array[5])]
  end
  return (gt_label,gt_box)
end


function readim(im_number)
  im = load(string(folder,"/dataset/train/images/",im_number))
  im = convert(Array{Int64},raw(im))
  if ndims(im)==2
    tmp = zeros(3,size(im,1),size(im,2))
    tmp[1,:,:] = im
    tmp[2,:,:] = im
    tmp[3,:,:] = im
    im = tmp
  end
  im = permutedims(im,[2 3 1])
  im = reshape(im,size(im,1),size(im,2),size(im,3),1)
  return convert(atype,im)
end

function create_classes()
  class = Dict{String,Int}()
  class["aeroplane"] = 1
  class["bicycle"] = 2
  class["bird"] = 3
  class["boat"] = 4
  class["bottle"] = 5
  class["bus"] = 6
  class["car"] = 7
  class["cat"] = 8
  class["chair"] = 9
  class["cow"] = 10
  class["diningtable"] = 11
  class["dog"] = 12
  class["horse"] = 13
  class["motorbike"] = 14
  class["person"] = 15
  class["pottedplant"] = 16
  class["sheep"] = 17
  class["sofa"] = 18
  class["train"] = 19
  class["tvmonitor"] = 20
  return class
end

function weights_rpn()
    w = Array(Any,16)
    # ZFNET
    w[1] = convert(atype,0.01.*randn(7,7,3,96))
    w[2] = convert(atype,zeros(1,1,96,1))
    w[3] = convert(atype,0.01.*randn(5,5,96,256))
    w[4] = convert(atype,zeros(1,1,256,1))
    w[5] = convert(atype,0.01.*randn(3,3,256,384))
    w[6] = convert(atype,zeros(1,1,384,1))
    w[7] = convert(atype,0.01.*randn(3,3,384,384))
    w[8] = convert(atype,zeros(1,1,384,1))
    w[9] = convert(atype,0.01.*randn(3,3,384,256))
    w[10] = convert(atype,zeros(1,1,256,1))
    # RPN
    w[11] = convert(atype,0.01.*randn(3,3,256,256))
    w[12] = convert(atype,zeros(1,1,256,1))
    w[13] = convert(atype,0.01.*randn(1,1,256,9))
    w[14] = convert(atype,zeros(1,1,9,1))
    w[15] = convert(atype,0.01.*randn(1,1,256,36))
    w[16] = convert(atype,zeros(1,1,36,1))

    return w
end

function forward_rpn(w,x)

    x = relu(conv4(w[1],x;padding=3,stride=2) .+ w[2])
		x = pool(x;window=3,padding=1,stride=2)
    x = relu(conv4(w[3],x;padding=2,stride=2) .+ w[4])
    x = pool(x;window=3,padding=1,stride=2)
    x = relu(conv4(w[5],x;padding=1,stride=1) .+ w[6])
    x = relu(conv4(w[7],x;padding=1,stride=1) .+ w[8])
    x = relu(conv4(w[9],x;padding=1,stride=1) .+ w[10])
    x = relu(conv4(w[11],x;padding=1,stride=1) .+ w[12])
    cls = conv4(w[13],x;padding=0,stride=1) .+ w[14]
    bbox = conv4(w[15],x;padding=0,stride=1) .+ w[16]

    return (cls,bbox)
end


epochs = 50
perc = 0.01 # percentage of the data to use
global imsize = 500
#global batchsize = 256
#global fg_fraction = 0.5
image_list = readdir("dataset/train/images")
label_list = readdir("dataset/train/annotations")

if gpu()>=0
	global atype = KnetArray{Float32}
  global folder = "/home/ec2-user/"
else
  global atype = Array{Float32}
  global folder = "/home/serkan/Documents/OKUL/COMP541/Project/PROJECT/"
end

global class = create_classes()
file = matopen("anchors.mat")
#anchors = convert(atype,read(file,"a"))
anchors = read(file,"a")
# for loop here
w = weights_rpn()
for epoch = 1:epochs
  if epoch < 10
    lr = 0.01
  else
    lr = 0.001
  end

  lss = 0
  for ind_im = 1:Int(round(length(label_list)*perc))
    @printf "%d " round(length(label_list)*perc)-ind_im
    #@printf "Processing image %d\n" ind_im
    (gt_label,gt_box) = readlabel(label_list[ind_im])
    (label_gold,bbox_gold) = prepare_roi(anchors,gt_box,gt_label)
    im = readim(image_list[ind_im])
    w=train(w,im,label_gold,bbox_gold,lr)
    lss += loss(w,im,label_gold,bbox_gold)
  end
  lss = lss/Int(round(length(label_list)*perc))
  @printf "\nepoch: %d, loss: %f\n" epoch lss
end
