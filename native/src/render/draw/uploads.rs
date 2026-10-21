//! What the drawer uploads for one view: the record buffers the draws of
//! a `Packet` read, as plain bytes. The drawer defines them because it is
//! the one that uploads them; the packer (pack/) fills them, and keeps
//! the records between frames.

/// Records to upload: their bytes, the bytes of one record, and the record
/// ranges written since the last upload (None: upload all of them).
pub struct Records<'a> {
    pub bytes: &'a [u8],
    pub record_size: usize,
    pub written: Option<Vec<(usize, usize)>>,
}

/// The record buffers a packet's draws read.
pub struct Uploads<'a> {
    /// Per page of vector records: objects, instances, fill vertices,
    /// fill colors, stroke points, stroke curves.
    pub vector_pages: Vec<[Records<'a>; 6]>,
    /// The mesh vertices and indices, when the packet has meshes.
    pub meshes: Option<[Records<'a>; 2]>,
}
