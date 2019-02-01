# the time axis output is jul_t=np.linspace(-maxlag,maglag,len(timeout)),which is centered at 0
# this is slightly different from the python output (by 1 sample)

using HDF5
using Random
using FFTW

const nx=36451
const nx2=72900
const ny=47
const FFTDIR = "/Users/zma/chengxin/KANTO/FFT/"
const CCFDIR = "/Users/zma/chengxin/KANTO/CCF/"
const maxlag=800
const downsamp_freq=20.0
const dt=1/downsamp_freq
const maxlag_sample = Int32(maxlag/dt)  #this should be 16000 for now
const nt = 2*maxlag_sample+1


function loadHDF5!(tmp1,tmp2,fft1,dat,name,std)
    HDF5.readarray(dat[name*".real"],HDF5.hdf5_type_id(Float32),tmp1)
    HDF5.readarray(dat[name*".imag"],HDF5.hdf5_type_id(Float32),tmp2)
    @. fft1=tmp1+1im*tmp2
    std.=read(attrs(dat[name*".real"]),"std")
end

# same as noise_module, n is the window half length
# for now, B[2]=mean(A[1]+A[2]+A[3]), if n=1; and B[1]=B[2]
# A and B are 1D Array
# NOTE: this operation can be O(length(A)), instead of O(length(A)*n)
function mov_avg!(A,n,B)
    nlen::Int32=0
    sum::Float32=0.
    n2::Int32=n*2+1
    if (length(A)<n2)
        println("error in mov_avg!!!")
        exit()
    end
    for i=1:n2-1
        sum+=abs(A[i])
    end
    i1::Int32=n2-1
    i2::Int32=1
    for i::Int32=n+1:length(A)-n
        i1+=1
        sum=sum+abs(A[i1])
        B[i]=sum/n2
        sum-=abs(A[i2])
        i2+=1
    end
    for i=1:n
        B[i]=abs(A[i])
    end
    for i=length(A)-n+1:length(A)
        B[i]=abs(A[i])
    end
end

function process(ifirst::Bool,outfile::String)

    time_ifft::Float32=0.
    time_mul::Float32=0.
    time_read::Float32=0.
    time_write::Float32=0.
    time_flip::Float32=0.
    time_source::Float32=0.
    n_ifft=0
    n_mul=0
    n_read=0
    n_write=0
    n_flip=0
    n_source=0


    if ifirst == false
        hdf5_out=h5open(outfile,"w")
        println("outputting to:",outfile)
    end

    allfiles=filter(x->x[end-2:end]==".h5",readdir(FFTDIR))
    println(allfiles)

    # doing in-place operation and pre-allocating array is key to make julia fast
    tmp1=Array{Float32}(undef,nx,ny)  #for load hdf5
    tmp2=similar(tmp1)
    fft_s=Array{Complex{Float32}}(undef,nx,ny)
    fft_s_smooth=Array{Float32}(undef,nx,ny)  #NOTE: this only smooth the abs 
    fft_r=similar(fft_s)
    fftback=Array{Complex{Float32}}(undef,nx)
    fftout=Array{Float32}(undef,nx2)
#    fftplan=plan_irfft(fftback,nx2)
    timeout=Array{Float32}(undef,nt)

    std_s=Array{Float64}(undef,ny)
    std_r=similar(std_s)
    ncount::Int32=0

    println("size:",sizeof(fft_s)/1024^2," ",sizeof(fft_s_smooth)/1024^2)

    for ii=1:length(allfiles)
        hdf5_s=h5open(FFTDIR*allfiles[ii],"r") 
        for name_s in filter(x->x[end-3:end]=="real",names(hdf5_s))
            name_s=name_s[1:end-5]
#            println("source:",name_s)
            tmp=split(name_s,"_")
            nsta_s=tmp[3]
            comp_s=tmp[4]

            t1=time()
            loadHDF5!(tmp1,tmp2,fft_s,hdf5_s,name_s,std_s)
#            println(std_s)
            for i=1:ny
                mov_avg!(view(fft_s,:,i),10,view(fft_s_smooth,:,i))
            end
            
            for j=1:ny,i=1:nx
                fft_s[i,j]=conj(fft_s[i,j])/fft_s_smooth[i,j]^2
                if isnan(fft_s[i,j])
                    println("NAN!!!",i,j,fft_s[i,j],fft_s_smooth[i,j])
                    exit()
                end
            end
            t2=time()
            n_source+=1
            time_source+=(t2-t1)
            
            # now loop over all receivers for all components
            for jj=1:length(allfiles)  #for easier comparison with python output 
#            for jj=ii+1:length(allfiles)  #you might want to use this for real run
                hdf5_r=h5open(FFTDIR*allfiles[jj],"r")
                nsta_r::String=split(names(hdf5_r)[1],"_")[3]

                for comp_r in ["HNE" "HNN" "HNU"]
                    tmp[3]=nsta_r
                    tmp[4]=comp_r
                    name_r=join(tmp,'_')
#                    println("receiver",name_r)

                    t1=time()
                    if !(name_r*".real" in names(hdf5_r))
                        continue
                    end
                    loadHDF5!(tmp1,tmp2,fft_r,hdf5_r,name_r,std_r)
                    t2=time()
#                    println("reading 1 receiver/comp",t2-t1)
                    n_read+=1
                    time_read+=(t2-t1)
                    
                    t1=time()
                    ncount=0
                    fftback.=0.
                    for j=1:ny
                        if std_r[j]>=10 || std_s[j]>=10 
                            continue
                        end
                        for i=1:nx           
                            fftback[i]+=fft_s[i,j]*fft_r[i,j]   #there must be a way to GPU this loop
                            if isnan(fftback[i])
                                println("mul error",i,j,"s:",fft_s[i,j],"r:",fft_r[i,j],"back",fftback[i])
                                exit()
                            end
                        end
                        ncount+=1
                        fftback[1]=0.
                    end
                    t2=time()
#                    println("multiplication:",t2-t1)
                    n_mul+=1
                    time_mul+=(t2-t1)
                    
                    t1=time()
                    @. fftback=fftback/ncount
                    t2=time()
#                    println("division",t2-t1,"ncount=",ncount)

                    t1=time()
                    fftout=irfft(fftback,nx2)   #is there a way to do insplace irfft??
                    t2=time()
#                    println("ifft",t2-t1)
                    n_ifft+=1
                    time_ifft+=(t2-t1)

                    t1=time()
                    j=maxlag_sample+1
                    for i=1:maxlag_sample+1
                        timeout[j]=fftout[i]
                        j+=1
                    end
                    j=length(fftout)-maxlag_sample+1
                    for i=1:maxlag_sample
                        timeout[i]=fftout[j]
                        j+=1
                    end
                    t2=time()
#                    println("flipping",t2-t1)
                    n_flip+=1
                    time_flip+=(t2-t1)

                    if ifirst   #THIS IS VERY LAMED>>>>>
                        return
                    end

                    t1=time()
                    write(hdf5_out,join((nsta_s,comp_s,nsta_r,comp_r),"_"),timeout)
                    t2=time()
#                    println("writing out",t2-t1)
                    n_write+=1
                    time_write+=(t2-t1)
                end           
                close(hdf5_r)
                

                

            end  #end of loop jj
        end  #end of looping over every datset in the source hdf5
        close(hdf5_s)
    end

    println("total in read:",n_read,' ',time_read)
    println("total in mul",n_mul,' ',time_mul)
    println("total in write",n_write,' ',time_write)
    println("total in ifft",n_ifft,' ',time_ifft)
    println("total in source",n_source,' ',time_source)
    println("total in flipping",n_flip,' ',time_flip)
end


outfile = CCFDIR*"test.h5"

# the strange thing for julia is that I need to run it first (so it compiles all functions???)
# with small dataset, then do a real run; there got to be an alternative way
@time process(true,outfile)
@time process(false,outfile)