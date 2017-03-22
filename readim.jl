using Images, FileIO

function minibatch(X, Y, bs=100)
	#takes raw input (X) and gold labels (Y)
	#returns list of minibatches (x, y)
	data = Any[]

	#start of step 1
	for i=1:round(Int,size(X,2)/bs)
		X_el = X[:,(i-1)*bs+1:i*bs]
		Y_el = Y[:,(i-1)*bs+1:i*bs]
		data_el = (X_el,Y_el)
		push!(data,data_el)
	end

	return data
end

function create_x(x_folder,imsize,n)
	x = zeros(Float32,500.^2.*3,1)
	image_list = readdir(x_folder)
	for i=1:n
		cur_im = load(string(x_folder,image_list[i]))
		cur_im = convert(Array{Float32},channelview(cur_im))
		if ndims(cur_im)==2
			tmp = zeros(3,size(cur_im,1),size(cur_im,2))
			tmp[1,:,:] = cur_im
			tmp[2,:,:] = cur_im
			tmp[3,:,:] = cur_im
			cur_im = tmp
		end
			cur_im = permutedims(cur_im,[2 3 1])
		# pad zeros to make image square
		if size(cur_im,2) < imsize
			cur_im = hcat(zeros(Float32,size(cur_im,1),Int64(floor((imsize-size(cur_im,2))/2)),size(cur_im,3)), cur_im, zeros(Float32,size(cur_im,1),Int64(ceil((imsize-size(cur_im,2))/2)),size(cur_im,3)))
		end
		if size(cur_im,1) < imsize
			cur_im = vcat(zeros(Float32,Int64(floor((imsize-size(cur_im,1))/2)),size(cur_im,2),size(cur_im,3)), cur_im, zeros(Float32,Int64(ceil((imsize-size(cur_im,1))/2)),size(cur_im,2),size(cur_im,3)))
		end
		cur_im_vec = vec(cur_im)
		x = hcat(x,cur_im_vec)
	end
	return x[:,2:end]
end

function create_dictionary(y_folder)
	label_list = readdir(y_folder)
	vocab = Dict{String,Int}()
	idx = 1
	for i=1:length(label_list)
		f = open(string(y_folder,label_list[i]));
		for ln in eachline(f)
			if !haskey(vocab,ln[1:end-1])
				vocab[ln[1:end-1]]=idx
				idx += 1
			end
		end
		close(f)
	end
	return vocab
end

function create_y(y_folder,vocab,n)
	label_list = readdir(y_folder)
	y = zeros(length(vocab),n)
	for i=1:n
		f = open(string(y_folder,label_list[i]));
		for ln in eachline(f)
			y[vocab[ln[1:end-1]],i]=1
		end
		close(f)
	end
	return y
end

function softmax_forward(data)
	yhat = zeros(20,size(data,2))
	for i = 1:size(data,2)
 		b = randn(20,1)
		w = randn(20,size(data,1))
		predict = w*data[:,i].+b
		prob = exp(predict)
		yhat[:,i] = prob./sum(prob,1);
	end
	return yhat
end

function cost(yhat,ygold)
	logprob = log(yhat)
	soft_loss = -sum(ygold.*logprob)/size(yhat,1)
end

xtrn_folder = "datasets/VOC2007/VOC2007_train/VOC2007/JPEGImages/"
xtst_folder = "datasets/VOC2007/VOC2007_test/VOC2007/JPEGImages/"
ytrn_folder = "datasets/VOC2007/VOC2007_train/VOC2007/Annotations/txt/"
ytst_folder = "datasets/VOC2007/VOC2007_test/VOC2007/Annotations/txt/"

n=10
imsize=500
vocab = create_dictionary(ytrn_folder)
xtrn = create_x(xtrn_folder,imsize,n)
xtst = create_x(xtst_folder,imsize,n)
ytrn = create_y(ytrn_folder,vocab,n)
ytst = create_y(ytst_folder,vocab,n)
bs = 2
trn_data = minibatch(xtrn, ytrn, bs)

ypred_trn = softmax_forward(xtrn)
loss = cost(ypred_trn,ytrn)
