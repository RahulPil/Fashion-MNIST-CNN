% create an image representation of a matrix
function plotbitmap(signal,ny,nx)
    size(signal)
    imagesc(reshape(signal,[nx,ny])')
end