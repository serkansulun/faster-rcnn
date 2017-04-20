using Images, Knet, ArgParse


function create_x(x_folder,imsize,bn,bs,n,atype,epoch)
	x = zeros(Float32,imsize,imsize,3,bs)
	image_list = readdir(x_folder)
	image_list = image_list[1:n]
	rng = srand(epoch)
	shuffle!(rng,image_list)
	if ceil(length(image_list)/bs) == bn
		last_idx = length(image_list)
	else
		last_idx = bn*bs
	end
	im=1

	for i=(bn-1)*bs+1:last_idx
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

    x[:,:,:,im] = round(cur_im.*255)
		im=im+1
	end
	x = convert(atype,x)
	return x
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
	return vocab, length(label_list)
end

function create_y(y_folder,vocab,bn,bs,n,atype,epoch)
	label_list = readdir(y_folder)
	label_list = label_list[1:n]
	rng = srand(epoch)
	shuffle!(rng,label_list)
	y = zeros(length(vocab),bs)
	if ceil(length(label_list)/bs) == bn
		last_idx = length(label_list)
	else
		last_idx = bn*bs
	end
	lab = 1

	for i=(bn-1)*bs+1:last_idx
		f = open(string(y_folder,label_list[i]));

		for ln in eachline(f)
			y[vocab[ln[1:end-1]],lab]=1
		end
		lab=lab+1
		close(f)
	end
	y = convert(atype,y)
	return y
end

function xavier(a...)
    w = rand(a...)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in 0.1.*randn")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end

function weights(atype)
    w = Array(Any,32)
    w[1] = 0.01.*randn(3,3,3,64)
    w[2] = zeros(1,1,64,1)
    w[3] = 0.01.*randn(3,3,64,64)
    w[4] = zeros(1,1,64,1)
    w[5] = 0.01.*randn(3,3,64,128)
    w[6] = zeros(1,1,128,1)
    w[7] = 0.01.*randn(3,3,128,128)
    w[8] = zeros(1,1,128,1)
    w[9] = 0.01.*randn(3,3,128,256)
    w[10] = zeros(1,1,256,1)
    w[11] = 0.01.*randn(3,3,256,256)
    w[12] = zeros(1,1,256,1)
    w[13] = 0.01.*randn(3,3,256,256)
    w[14] = zeros(1,1,256,1)
    w[15] = 0.01.*randn(3,3,256,512)
    w[16] = zeros(1,1,512,1)
    w[17] = 0.01.*randn(3,3,512,512)
    w[18] = zeros(1,1,512,1)
    w[19] = 0.01.*randn(3,3,512,512)
    w[20] = zeros(1,1,512,1)
    w[21] = 0.01.*randn(3,3,512,512)
    w[22] = zeros(1,1,512,1)
    w[23] = 0.01.*randn(3,3,512,512)
    w[24] = zeros(1,1,512,1)
    w[25] = 0.01.*randn(3,3,512,512)
    w[26] = zeros(1,1,512,1)
    w[27] = 0.01.*randn(4096,7*7*512)
    w[28] = zeros(4096,1)
    w[29] = 0.01.*randn(4096,4096)
    w[30] = zeros(4096,1)
    w[31] = 0.01.*randn(20,4096)
    w[32] = zeros(20,1)
    return map(a->convert(atype,a), w)
end

function weightsRPN
	w[1] = 0.1.*randn(3,3,384,256)
    w[2] = zeros(1,1,256,1)
    w[3] = 0.1.*randn(1,1,256,18)
    w[4] = zeros(1,1,18,1)
    w[5] = 0.1.*randn(1,1,256,26)
    w[6] = zeros(1,1,36,1)
    
end


# function initprms(w)
# 	prms = Array(Any,length(w))
# 	for i=1:length(w)
# 		prms[i] = Adam()
# 	end
# 	return prms
# end

function predict(w,x)

    x = relu(conv4(w[1],x;padding=1) .+ w[2])
  	x = relu(conv4(w[3],x;padding=1) .+ w[4])
		x = pool(x)
		x = relu(conv4(w[5],x;padding=1) .+ w[6])
  	x = relu(conv4(w[7],x;padding=1) .+ w[8])
		x = pool(x)
		x = relu(conv4(w[9],x;padding=1) .+ w[10])
  	x = relu(conv4(w[11],x;padding=1) .+ w[12])
		x = relu(conv4(w[13],x;padding=1) .+ w[14])
		x = pool(x)
		x = relu(conv4(w[15],x;padding=1) .+ w[16])
		x = relu(conv4(w[17],x;padding=1) .+ w[18])
		x = relu(conv4(w[19],x;padding=1) .+ w[20])
		x = pool(x)
		x = relu(conv4(w[21],x;padding=1) .+ w[22])
		x = relu(conv4(w[23],x;padding=1) .+ w[24])
		x = relu(conv4(w[25],x;padding=1) .+ w[26])
		x = pool(x)

    x = mat(x)
		x = relu(w[27]*x .+ w[28])
		x = relu(w[29]*x .+ w[30])
		x = relu(w[31]*x .+ w[32])

    return x
end

function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm =  ypred .- log(sum(exp(ypred),1)) #logp(ypred,1)  #
    return -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function train(w, x, y, lr)#; iters=1800)
            g = lossgradient(w, x, y)
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

function accuracy(w,x,y,threshold)

      ypred = predict(w,x)
      ncorrect = sum(y .* (ypred .> threshold))
      nobj = sum(y)
			ninstance = size(y,2)
      nloss = -sum(y.*logp(ypred,1))

    return (nloss/ninstance,ncorrect/nobj)
end

function main(args=ARGS)
	s = ArgParseSettings()
    s.description="FASTER RCNN"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--bs"; arg_type=Int; default=10; help="minibatch size")
				("--threshold"; arg_type=Float64; default=0.5; help="threshold")
				("--samplesize"; arg_type=Int; default=0; help="sample size")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
        ("--epochs"; arg_type=Int; default=30; help="number of epochs for training")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
if gpu()>=0
	atype = KnetArray{Float32}
else
  atype = Array{Float32}
end


imsize=224
threshold = 0.5
xtrn_folder = "dataset/VOC2007/VOC2007_train/JPEGImages224/"
xtst_folder = "dataset/VOC2007/VOC2007_test/JPEGImages224/"
ytrn_folder = "dataset/VOC2007/VOC2007_train/Annotations/txt/"
ytst_folder = "dataset/VOC2007/VOC2007_test/Annotations/txt/"

vocab,n = create_dictionary(ytrn_folder)

if o[:samplesize]>0
	n = o[:samplesize]
end
batch_count = ceil(Int,n/o[:bs])

w = weights(atype)

#prms = initprms(w)
#report(epoch)=println(:epoch,epoch,:trn,accuracy(w,xtrn,threshold))#,:tst,accuracy(w,dtst)))
#report(0)
# Main part of our training process
@time for epoch=1:o[:epochs]
	@printf "\nRemaining batches: "
	lss = Float32(0)
	acc = Float32(0)

	for batch=1:batch_count

		@printf "%d " batch_count-batch
		xtrn = create_x(xtrn_folder,imsize,batch,o[:bs],n,atype,epoch)
		#xtst = create_x(xtst_folder,imsize,n)
		ytrn = create_y(ytrn_folder,vocab,batch,o[:bs],n,atype,epoch)
		#ytst = create_y(ytst_folder,vocab,n)
		lss_init,acc_init = accuracy(w,xtrn,ytrn,o[:threshold])
		#@printf "\nBatch loss: %f, Batch accuracy: %f" lss_init acc_init
		train(w, xtrn,ytrn, o[:lr])
		lssnew,accnew = accuracy(w,xtrn,ytrn,o[:threshold])
		lss+=loss(w,xtrn,ytrn)
		acc+=accnew
		#println(:epoch ,epoch,:batch,batch,:trn,accuracy(w,xtrn,ytrn,threshold))
	end
	@printf "\nepoch: %d, loss: %f, accuracy: %f" epoch lss/batch_count acc/batch_count
	@printf "\n"
	if o[:gcheck] > 0
      gradcheck(loss, w, first(dtrn)...; gcheck=o[:gcheck], verbose=true)
	end

		#report(epoch)
end

return w
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "rcnn.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end
