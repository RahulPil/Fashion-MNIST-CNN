function plotbitmap(signal,ny,nx)
    size(signal)
    imagesc(reshape(signal,[nx,ny])')
    
end