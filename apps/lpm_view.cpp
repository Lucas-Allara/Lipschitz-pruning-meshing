// Interactive viewer of the dynamic update, drawn with Vulkan from the buffers where the GPU
// computes the mesh: GpuPruner::dynamic_frame_gpu writes the vertices and the triangles, and the
// vertex shader (shaders/view_mesh.vert) reads them there, so the mesh never leaves the GPU.
// SDL2 opens the window.
//
//   lpm_view [trees|monument|molecule|FILE.json] [depth] [--vsync] [--bench N] [--dc] [--coarse]
//            [--colors] [--edges] [--vertices]
// The left button on the sphere drags it, elsewhere it orbits the camera; the right button and the
// wheel zoom, the middle button pans. Keys 1, 2, 3 load Trees, Monument and Molecule, the left and
// right arrows go through every scene (then the CAD parts of scenes/inversecsg, when they are
// there), + and - (or the up and down arrows) change the depth, space moves the sphere around the
// scene, Esc quits. M switches between marching cubes on the leaves (the default) and dual
// contouring (--dc starts with it); G between those uniform leaves and coarse leaves, by dual
// contouring, where each frame meshes the coarse leaves that the sphere reaches by their finest
// leaves (--coarse starts with them). C colors the mesh by the size of the pruned tree of each
// vertex, E draws the edges of the triangles and V the vertices (--colors, --edges and --vertices
// start with them on); the title shows what is on. --bench N moves the sphere along that path for
// N frames, prints the frame times and quits.
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "lpm/gpu.h"
#include "lpm/threads.h"

using namespace lpm;

namespace {

#define VK_CHECK(x)                                                                                              \
    do {                                                                                                         \
        VkResult r_ = (x);                                                                                       \
        if (r_ != VK_SUCCESS) throw std::runtime_error("Vulkan error " + std::to_string(r_) + " in " #x);       \
    } while (0)

const uint32_t kMeshVert[] =
#include "view_mesh_vert.inc"
    ;
const uint32_t kMeshFrag[] =
#include "view_mesh_frag.inc"
    ;

constexpr double kRadius = 0.2, kK = 0.1, kMargin = 1e-3, kFov = 30.0 * M_PI / 180.0;
constexpr double kCoarseTol = 0.005;  // coarse leaves: residual of the quadric relative to the cell, as the CAD table of the poster

struct Vec3 {
    double x = 0, y = 0, z = 0;
};
Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
double norm(Vec3 a) { return std::sqrt(dot(a, a)); }
Vec3 normalize(Vec3 a) { return a * (1.0 / norm(a)); }
// Rotation of v by angle a around the unit axis k (Rodrigues).
Vec3 rotate(Vec3 v, Vec3 k, double a) {
    return v * std::cos(a) + cross(k, v) * std::sin(a) + k * (dot(k, v) * (1.0 - std::cos(a)));
}

// Column-major 4 x 4 matrices, as GLSL reads them.
struct M4 {
    float m[16] = {};
};
M4 mul(const M4& a, const M4& b) {
    M4 r;
    for (int c = 0; c < 4; c++)
        for (int w = 0; w < 4; w++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += a.m[k * 4 + w] * b.m[c * 4 + k];
            r.m[c * 4 + w] = s;
        }
    return r;
}
M4 look_at(Vec3 eye, Vec3 focal, Vec3 up) {
    const Vec3 f = normalize(focal - eye), s = normalize(cross(f, up)), u = cross(s, f);
    M4 m;
    m.m[0] = (float)s.x; m.m[4] = (float)s.y; m.m[8] = (float)s.z;
    m.m[1] = (float)u.x; m.m[5] = (float)u.y; m.m[9] = (float)u.z;
    m.m[2] = (float)-f.x; m.m[6] = (float)-f.y; m.m[10] = (float)-f.z;
    m.m[12] = (float)-dot(s, eye); m.m[13] = (float)-dot(u, eye); m.m[14] = (float)dot(f, eye);
    m.m[15] = 1.0f;
    return m;
}
// Vulkan clip space: y down, depth from 0 (near) to 1 (far).
M4 perspective(double fovy, double aspect, double n, double f) {
    const double t = 1.0 / std::tan(0.5 * fovy);
    M4 m;
    m.m[0] = (float)(t / aspect);
    m.m[5] = (float)-t;
    m.m[10] = (float)(f / (n - f));
    m.m[11] = -1.0f;
    m.m[14] = (float)(n * f / (n - f));
    return m;
}

// The camera of scripts/render_meshes.py (VIEWS), a trackball around the focal point.
struct Camera {
    Vec3 eye, focal, up{0, 1, 0};
    Vec3 forward() const { return normalize(focal - eye); }
    Vec3 right() const { return normalize(cross(forward(), up)); }
    void orbit(double dx, double dy) {
        const double k = 0.3 * M_PI / 180.0;  // radians per pixel
        Vec3 d = rotate(eye - focal, normalize(up), -dx * k);
        const Vec3 r = normalize(cross(d * -1.0, up));
        d = rotate(d, r, -dy * k);
        up = rotate(up, r, -dy * k);
        eye = focal + d;
        up = normalize(cross(right(), forward()));
    }
    void dolly(double factor) { eye = focal + (eye - focal) * factor; }
    void pan(double dx, double dy, double height) {
        const double s = 2.0 * norm(eye - focal) * std::tan(0.5 * kFov) / height;
        const Vec3 u = cross(right(), forward());
        const Vec3 t = right() * (-dx * s) + u * (dy * s);
        eye = eye + t;
        focal = focal + t;
    }
    // Ray through the window point (px, py) of a w x h window.
    Vec3 ray(double px, double py, double w, double h) const {
        const double t = std::tan(0.5 * kFov);
        const double x = (2.0 * px / w - 1.0) * t * w / h, y = (1.0 - 2.0 * py / h) * t;
        const Vec3 f = forward(), r = right(), u = cross(r, f);
        return normalize(f + r * x + u * y);
    }
};

struct PushConstants {
    float mvp[16];
    float eye[4];
    uint64_t static_v, new_v, faces;
    uint32_t nv_static, flags;    // flags: 1 colors, 2 edges, 4 vertices, 8 large points
    uint64_t static_t, new_t;     // tree sizes of the vertices
};
static_assert(sizeof(PushConstants) == 128, "layout of PC in shaders/view_common.glsl");

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

// Swapchain, multisampled attachments and the pipeline of the mesh; draws from the GPU mesh buffers.
struct Renderer {
    // The swapchain and the pipeline go before the device, also when an exception ends main.
    ~Renderer() { destroy(); }

    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    bool vsync = false;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{0, 0};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkSemaphore> done;  // per swapchain image: rendering finished, waited by the presentation
    Image color, depth;             // multisampled
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    const VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline mesh_pipe = VK_NULL_HANDLE, line_pipe = VK_NULL_HANDLE, point_pipe = VK_NULL_HANDLE;
    bool large_points = false;
    VkPhysicalDeviceMemoryProperties memprops{};

    void init(const GpuHandles& h, VkSurfaceKHR s, bool vs, uint32_t w, uint32_t hh) {
        inst = (VkInstance)h.instance;
        phys = (VkPhysicalDevice)h.physical_device;
        dev = (VkDevice)h.device;
        queue = (VkQueue)h.queue;
        qfam = h.queue_family;
        surface = s;
        vsync = vs;
        VkBool32 ok = VK_FALSE;
        VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(phys, qfam, surface, &ok));
        if (!ok) throw std::runtime_error("the compute queue cannot present to this window");
        vkGetPhysicalDeviceMemoryProperties(phys, &memprops);
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys, &props);
        const VkSampleCountFlags sc = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
        samples = (sc & VK_SAMPLE_COUNT_4_BIT) ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT;

        uint32_t n = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(n);
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n, formats.data());
        format = formats.at(0).format;
        for (const VkSurfaceFormatKHR& f : formats)
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) format = f.format;

        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = qfam;
        VK_CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fence));
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(dev, &sci, nullptr, &acquired));

        VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants)};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &range;
        VK_CHECK(vkCreatePipelineLayout(dev, &lci, nullptr, &layout));
        // The triangles go a little back in depth, so that their edges and vertices show in front.
        mesh_pipe = pipeline(kMeshVert, sizeof(kMeshVert), kMeshFrag, sizeof(kMeshFrag), VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true);
        line_pipe = pipeline(kMeshVert, sizeof(kMeshVert), kMeshFrag, sizeof(kMeshFrag), VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false);
        point_pipe = pipeline(kMeshVert, sizeof(kMeshVert), kMeshFrag, sizeof(kMeshFrag), VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false);
        VkPhysicalDeviceFeatures feats;
        vkGetPhysicalDeviceFeatures(phys, &feats);
        large_points = feats.largePoints;  // GpuPruner enables it on graphics devices that have it
        create_swapchain(w, hh);
    }

    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) const {
        for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want) return i;
        throw std::runtime_error("no suitable memory type");
    }

    Image make_image(VkFormat f, VkImageUsageFlags usage, VkImageAspectFlags aspect) const {
        Image im;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = f;
        ici.extent = {extent.width, extent.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = samples;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(dev, &ici, nullptr, &im.image));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(dev, im.image, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &im.mem));
        VK_CHECK(vkBindImageMemory(dev, im.image, im.mem, 0));
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = im.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = f;
        vci.subresourceRange = {aspect, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(dev, &vci, nullptr, &im.view));
        return im;
    }
    void free_image(Image& im) const {
        if (im.view) vkDestroyImageView(dev, im.view, nullptr);
        if (im.image) vkDestroyImage(dev, im.image, nullptr);
        if (im.mem) vkFreeMemory(dev, im.mem, nullptr);
        im = Image{};
    }

    VkShaderModule shader(const uint32_t* code, size_t bytes) const {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m;
        VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m));
        return m;
    }

    // The mesh, with depth test and write (the moving sphere is part of it), as triangles, lines or
    // points; bias pushes the triangles back. The lines and points blend over the triangles without
    // writing depth, so that where they crowd (far away) the mesh shows through.
    VkPipeline pipeline(const uint32_t* vs, size_t vs_bytes, const uint32_t* fs, size_t fs_bytes, VkPrimitiveTopology topology,
                        bool bias) const {
        VkShaderModule mv = shader(vs, vs_bytes), mf = shader(fs, fs_bytes);
        VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module = mv;
        st[0].pName = "main";
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = mf;
        st[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = topology;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        rs.depthBiasEnable = bias ? VK_TRUE : VK_FALSE;
        rs.depthBiasConstantFactor = bias ? 2.0f : 0.0f;
        rs.depthBiasSlopeFactor = bias ? 1.5f : 0.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = samples;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
            cba.blendEnable = VK_TRUE;
            cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cba.colorBlendOp = VK_BLEND_OP_ADD;
            cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            cba.alphaBlendOp = VK_BLEND_OP_ADD;
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 2;
        dy.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        ri.colorAttachmentCount = 1;
        ri.pColorAttachmentFormats = &format;
        ri.depthAttachmentFormat = depth_format;
        VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pci.pNext = &ri;
        pci.stageCount = 2;
        pci.pStages = st;
        pci.pVertexInputState = &vi;
        pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vp;
        pci.pRasterizationState = &rs;
        pci.pMultisampleState = &ms;
        pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &cb;
        pci.pDynamicState = &dy;
        pci.layout = layout;
        VkPipeline p;
        VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &p));
        vkDestroyShaderModule(dev, mv, nullptr);
        vkDestroyShaderModule(dev, mf, nullptr);
        return p;
    }

    void create_swapchain(uint32_t w, uint32_t h) {
        VkSurfaceCapabilitiesKHR caps;
        VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, &caps));
        if (caps.currentExtent.width != UINT32_MAX) extent = caps.currentExtent;
        else extent = {std::clamp(w, caps.minImageExtent.width, caps.maxImageExtent.width),
                       std::clamp(h, caps.minImageExtent.height, caps.maxImageExtent.height)};
        if (extent.width == 0 || extent.height == 0) return;  // minimized
        uint32_t n = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &n, nullptr);
        std::vector<VkPresentModeKHR> modes(n);
        vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &n, modes.data());
        auto has = [&](VkPresentModeKHR m) { return std::find(modes.begin(), modes.end(), m) != modes.end(); };
        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        if (!vsync) {
            if (has(VK_PRESENT_MODE_MAILBOX_KHR)) mode = VK_PRESENT_MODE_MAILBOX_KHR;
            else if (has(VK_PRESENT_MODE_IMMEDIATE_KHR)) mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
        uint32_t count = std::max(caps.minImageCount + 1, 3u);
        if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);
        VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        sci.surface = surface;
        sci.minImageCount = count;
        sci.imageFormat = format;
        sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        sci.imageExtent = extent;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = caps.currentTransform;
        sci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                                                                               : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        sci.presentMode = mode;
        sci.clipped = VK_TRUE;
        sci.oldSwapchain = swapchain;
        VkSwapchainKHR fresh;
        VK_CHECK(vkCreateSwapchainKHR(dev, &sci, nullptr, &fresh));
        destroy_swapchain();
        swapchain = fresh;
        vkGetSwapchainImagesKHR(dev, swapchain, &n, nullptr);
        images.resize(n);
        vkGetSwapchainImagesKHR(dev, swapchain, &n, images.data());
        for (VkImage im : images) {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = im;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageView v;
            VK_CHECK(vkCreateImageView(dev, &vci, nullptr, &v));
            views.push_back(v);
            VkSemaphoreCreateInfo smi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VkSemaphore s;
            VK_CHECK(vkCreateSemaphore(dev, &smi, nullptr, &s));
            done.push_back(s);
        }
        color = make_image(format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        depth = make_image(depth_format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT);
        fprintf(stderr, "swapchain %ux%u, %u images, %s, %dx MSAA\n", extent.width, extent.height, n,
                mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo (vsync)",
                (int)samples);
    }

    // Everything of the swapchain but the swapchain itself, which create_swapchain retires.
    void destroy_swapchain() {
        for (VkImageView v : views) vkDestroyImageView(dev, v, nullptr);
        for (VkSemaphore s : done) vkDestroySemaphore(dev, s, nullptr);
        views.clear();
        done.clear();
        images.clear();
        free_image(color);
        free_image(depth);
        if (swapchain) vkDestroySwapchainKHR(dev, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }

    static void image_barrier(VkCommandBuffer c, VkImage im, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
                              VkPipelineStageFlags src, VkAccessFlags src_access, VkPipelineStageFlags dst, VkAccessFlags dst_access) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;
        b.newLayout = to;
        b.srcAccessMask = src_access;
        b.dstAccessMask = dst_access;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im;
        b.subresourceRange = {aspect, 0, 1, 0, 1};
        vkCmdPipelineBarrier(c, src, dst, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    // Draws the mesh and presents; false when the swapchain must be recreated.
    // Returns after the GPU has finished, so the next dynamic frame may overwrite the buffers.
    bool draw(PushConstants pc, uint32_t n_triangles, bool edges, bool vertices) {
        if (!swapchain) return false;
        uint32_t idx = 0;
        VkResult r = vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &idx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) return false;
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) VK_CHECK(r);

        VK_CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        // The mesh written by the compute submissions of the dynamic update, read by the vertex shader.
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        const VkPipelineStageFlags out_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        image_barrier(cmd, images[idx], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      out_stage, 0, out_stage, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        image_barrier(cmd, color.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      out_stage, 0, out_stage, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        const VkPipelineStageFlags tests = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        image_barrier(cmd, depth.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                      tests, 0, tests, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT);

        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        ca.imageView = samples == VK_SAMPLE_COUNT_1_BIT ? views[idx] : color.view;
        ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        if (samples != VK_SAMPLE_COUNT_1_BIT) {
            ca.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            ca.resolveImageView = views[idx];
            ca.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.storeOp = samples == VK_SAMPLE_COUNT_1_BIT ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ca.clearValue.color = {{1.0f, 1.0f, 1.0f, 1.0f}};
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        da.imageView = depth.view;
        da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        da.clearValue.depthStencil = {1.0f, 0};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, extent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &ca;
        ri.pDepthAttachment = &da;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{0.0f, 0.0f, (float)extent.width, (float)extent.height, 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, extent};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        if (n_triangles) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipe);
            vkCmdDraw(cmd, 3 * n_triangles, 1, 0, 0);
            const uint32_t colors = pc.flags & 1u;
            if (edges) {
                pc.flags = colors | 2u;
                vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipe);
                vkCmdDraw(cmd, 6 * n_triangles, 1, 0, 0);
            }
            if (vertices) {
                pc.flags = colors | 4u | (large_points ? 8u : 0u);
                vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point_pipe);
                vkCmdDraw(cmd, 3 * n_triangles, 1, 0, 0);
            }
        }
        vkCmdEndRendering(cmd);
        image_barrier(cmd, images[idx], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      out_stage, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &acquired;
        si.pWaitDstStageMask = &out_stage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &done[idx];
        VK_CHECK(vkResetFences(dev, 1, &fence));
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &done[idx];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &idx;
        r = vkQueuePresentKHR(queue, &pi);
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) return false;
        VK_CHECK(r);
        return true;
    }

    void destroy() {
        if (!dev) return;
        vkDeviceWaitIdle(dev);
        destroy_swapchain();
        for (VkPipeline p : {mesh_pipe, line_pipe, point_pipe}) if (p) vkDestroyPipeline(dev, p, nullptr);
        if (layout) vkDestroyPipelineLayout(dev, layout, nullptr);
        if (acquired) vkDestroySemaphore(dev, acquired, nullptr);
        if (fence) vkDestroyFence(dev, fence, nullptr);
        if (pool) vkDestroyCommandPool(dev, pool, nullptr);
        if (surface) vkDestroySurfaceKHR(inst, surface, nullptr);
        dev = VK_NULL_HANDLE;
    }
};

struct View {
    const char* name;
    Vec3 position, focal;
};
const View kViews[] = {{"trees", {2.6, 1.9, 3.2}, {0.0, 0.7, 0.15}},
                       {"monument", {3.3, 2.5, 3.5}, {0.0, 0.25, 0.0}},
                       {"molecule", {2.2, 1.8, 2.6}, {0.0, 0.0, 0.0}}};
const char* kScenes[] = {"trees", "monument", "molecule"};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    lpm::passive_omp_wait(argv);
    std::string scene_arg = "trees";
    int depth = 7, bench = 0, positional = 0;
    bool vsync = false, use_mc = true, colors = false, edges = false, vertices = false, coarse = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--vsync") vsync = true;
        else if (a == "--bench" && i + 1 < argc) bench = atoi(argv[++i]);
        else if (a == "--dc") use_mc = false;
        else if (a == "--colors") colors = true;
        else if (a == "--edges") edges = true;
        else if (a == "--vertices") vertices = true;
        else if (a == "--coarse") coarse = true;
        else if (a.rfind("--", 0) == 0) {
            fprintf(stderr, "usage: lpm_view [trees|monument|molecule|FILE.json] [depth] [--vsync] [--bench N] [--dc] [--colors] [--edges]"
                            " [--vertices] [--coarse]\n");
            return 1;
        } else if (positional++ == 0) scene_arg = a;
        else depth = atoi(a.c_str());
    }
    if (depth < 1 || depth > 9) {
        fprintf(stderr, "the depth must be between 1 and 9 (dense grids on the GPU)\n");
        return 1;
    }
    std::string root = ".";
    {
        // The scenes are found next to the binary (build/..) or in the current directory.
        std::string exe = argv[0];
        const size_t slash = exe.rfind('/');
        if (slash != std::string::npos) root = exe.substr(0, slash) + "/..";
    }
    auto scene_path = [&](const std::string& s) {
        if (s.size() > 5 && s.substr(s.size() - 5) == ".json") return s;
        return root + "/scenes/" + s + ".json";
    };

    SDL_SetHint(SDL_HINT_VIDEODRIVER, "wayland,x11");
    if (SDL_Init(SDL_INIT_VIDEO) != 0) { fprintf(stderr, "SDL: %s\n", SDL_GetError()); return 1; }
    SDL_Window* win = SDL_CreateWindow("lpm_view", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1500, 1000,
                                       SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!win) { fprintf(stderr, "SDL: %s\n", SDL_GetError()); return 1; }
    unsigned n_ext = 0;
    SDL_Vulkan_GetInstanceExtensions(win, &n_ext, nullptr);
    std::vector<const char*> ext(n_ext);
    SDL_Vulkan_GetInstanceExtensions(win, &n_ext, ext.data());
    int status = 0;
    try {
        std::unique_ptr<GpuPruner> gpu(new GpuPruner(std::vector<std::string>(ext.begin(), ext.end()), true));
        const GpuHandles h = gpu->handles();
        VkSurfaceKHR surface;
        if (!SDL_Vulkan_CreateSurface(win, (VkInstance)h.instance, &surface)) throw std::runtime_error(SDL_GetError());
        int dw = 0, dh = 0;
        SDL_Vulkan_GetDrawableSize(win, &dw, &dh);
        Renderer rd;
        rd.init(h, surface, vsync, (uint32_t)dw, (uint32_t)dh);
        fprintf(stderr, "%s, %s\n", gpu->device_name().c_str(), SDL_GetCurrentVideoDriver());

        Scene base;
        DynamicScene dyn;
        Camera cam;
        Vec3 center;
        std::string scene_name;
        GpuMeshBuffers mesh;
        GpuFrameStats fs;
        // A scene, meshed with marching cubes or dual contouring; keep_view keeps the camera and the
        // sphere (switching the method).
        // Every scene: the three of the paper, then the CAD parts (local only, see scenes/inversecsg).
        std::vector<std::string> scenes(std::begin(kScenes), std::end(kScenes));
        {
            std::vector<std::string> cad;
            std::error_code ec;
            for (const auto& entry : std::filesystem::directory_iterator(root + "/scenes/inversecsg", ec))
                if (entry.path().extension() == ".json") cad.push_back(entry.path().string());
            std::sort(cad.begin(), cad.end());
            scenes.insert(scenes.end(), cad.begin(), cad.end());
        }
        // The scene of the command line is the one of the list that names the same file, or goes last,
        // so that the keys 1, 2 and 3 keep Trees, Monument and Molecule.
        auto same_scene = [&](const std::string& a, const std::string& b) {
            std::error_code ea, eb;
            const std::filesystem::path pa = std::filesystem::weakly_canonical(scene_path(a), ea),
                                        pb = std::filesystem::weakly_canonical(scene_path(b), eb);
            return ea || eb ? a == b : pa == pb;
        };
        size_t scene_idx =
            std::find_if(scenes.begin(), scenes.end(), [&](const std::string& s) { return same_scene(s, scene_arg); }) - scenes.begin();
        if (scene_idx == scenes.size()) scenes.push_back(scene_arg);
        // The dynamic update of the current mode: uniform leaves by marching cubes or dual contouring,
        // or coarse leaves by dual contouring (marching cubes would leave cracks between leaves of
        // different sizes), where each frame meshes the coarse leaves that the sphere reaches by their
        // finest leaves.
        auto mode_name = [&]() { return coarse ? "dual contouring, coarse leaves" : use_mc ? "marching cubes" : "dual contouring"; };
        auto setup = [&]() {
            Counters c;
            DCParams dp;
            dp.marching_cubes = use_mc && !coarse;
            CoarseParams cp;
            cp.tol = kCoarseTol;
            gpu->dynamic_setup(dyn, depth, kMargin, dp, c, coarse ? &cp : nullptr);
        };
        auto load = [&](const std::string& s, bool keep_view = false) {
            vkDeviceWaitIdle((VkDevice)h.device);
            Timer t;
            base = load_scene_json(scene_path(s));
            dyn = make_dynamic_scene(base, kK, kRadius);
            const std::string name = std::filesystem::path(s).stem().string();
            if (!keep_view) {
                scene_name = name;
                const View* v = nullptr;
                for (const View& x : kViews) if (name == x.name) v = &x;
                const Vec3 lo{base.aabb_min[0], base.aabb_min[1], base.aabb_min[2]}, hi{base.aabb_max[0], base.aabb_max[1], base.aabb_max[2]};
                cam = Camera{};
                cam.focal = v ? v->focal : (lo + hi) * 0.5;
                cam.eye = v ? v->position : cam.focal + Vec3{1.0, 0.75, 1.1} * (0.8 * norm(hi - lo));
                center = cam.focal;
            }
            setup();
            fprintf(stderr, "%s, depth %d, %s: setup %.2f s\n", name.c_str(), depth, mode_name(), t.seconds());
        };
        // The first frame of a setup allocates the buffers of the frames, which may not fit either.
        auto first_frame = [&]() {
            const double c[3] = {center.x, center.y, center.z};
            Counters cnt;
            mesh = gpu->dynamic_frame_gpu(c, cnt, fs);
        };
        // A change of scene, depth, method or leaves from a key; when the new one does not fit (device
        // memory, at depth 9), the previous one comes back, with its camera and sphere.
        auto change = [&](size_t new_idx, int new_depth, bool new_mc, bool new_coarse, bool keep_view) {
            const size_t old_idx = scene_idx;
            const int old_depth = depth;
            const bool old_mc = use_mc, old_coarse = coarse;
            const Camera old_cam = cam;
            const Vec3 old_center = center;
            const std::string old_name = scene_name;
            scene_idx = new_idx;
            depth = new_depth;
            use_mc = new_mc;
            coarse = new_coarse;
            try {
                load(scenes[scene_idx], keep_view);
                first_frame();
            } catch (const std::exception& ex) {
                fprintf(stderr, "lpm_view: %s, depth %d, %s: %s\n", scenes[scene_idx].c_str(), depth, mode_name(), ex.what());
                gpu->release_dynamic();
                scene_idx = old_idx;
                depth = old_depth;
                use_mc = old_mc;
                coarse = old_coarse;
                cam = old_cam;
                center = old_center;
                scene_name = old_name;
                load(scenes[scene_idx], true);
                first_frame();
            }
        };
        try {
            load(scenes[scene_idx]);
            first_frame();
        } catch (const std::exception& ex) {
            if (!coarse) throw;
            fprintf(stderr, "lpm_view: %s, depth %d, %s: %s\n", scenes[scene_idx].c_str(), depth, mode_name(), ex.what());
            gpu->release_dynamic();
            coarse = false;  // the uniform leaves of the dynamic update, then
            load(scenes[scene_idx]);
            first_frame();
        }

        bool running = true, dirty = true, redraw = true, animate = bench > 0, dragging_sphere = false;
        int button = 0;
        Vec3 grab_offset;
        double path_t = 0.0;
        const int path_frames = bench > 0 ? bench : 240;
        int frame = 0;
        std::vector<double> t_frame, t_update, t_draw, t_descent, t_region, t_dc;
        double title_clock = 0.0;
        Timer clock;
        uint32_t frames_since_title = 0;
        while (running) {
            SDL_Event e;
            const bool idle = !animate && !dirty && !redraw;
            if (idle && SDL_WaitEvent(nullptr) == 0) break;
            while (SDL_PollEvent(&e)) {
                int ww = 1, wh = 1;
                SDL_GetWindowSize(win, &ww, &wh);
                switch (e.type) {
                case SDL_QUIT: running = false; break;
                case SDL_WINDOWEVENT:
                    if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED || e.window.event == SDL_WINDOWEVENT_EXPOSED) {
                        if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                            vkDeviceWaitIdle(rd.dev);
                            SDL_Vulkan_GetDrawableSize(win, &dw, &dh);
                            rd.create_swapchain((uint32_t)dw, (uint32_t)dh);
                        }
                        redraw = true;
                    }
                    break;
                case SDL_KEYDOWN:
                    if (e.key.keysym.sym == SDLK_ESCAPE || e.key.keysym.sym == SDLK_q) running = false;
                    else if (e.key.keysym.sym == SDLK_SPACE) animate = !animate;
                    else if (e.key.keysym.sym >= SDLK_1 && e.key.keysym.sym <= SDLK_3) {
                        change((size_t)(e.key.keysym.sym - SDLK_1), depth, use_mc, coarse, false);
                        dirty = redraw = true;
                    } else if (e.key.keysym.sym == SDLK_RIGHT || e.key.keysym.sym == SDLK_LEFT) {
                        const size_t n = scenes.size();
                        change((scene_idx + (e.key.keysym.sym == SDLK_RIGHT ? 1 : n - 1)) % n, depth, use_mc, coarse, false);
                        dirty = redraw = true;
                    } else if (e.key.keysym.sym == SDLK_PLUS || e.key.keysym.sym == SDLK_KP_PLUS || e.key.keysym.sym == SDLK_EQUALS ||
                               e.key.keysym.sym == SDLK_UP || e.key.keysym.sym == SDLK_MINUS || e.key.keysym.sym == SDLK_KP_MINUS ||
                               e.key.keysym.sym == SDLK_DOWN) {
                        const bool up = e.key.keysym.sym == SDLK_PLUS || e.key.keysym.sym == SDLK_KP_PLUS || e.key.keysym.sym == SDLK_EQUALS ||
                                        e.key.keysym.sym == SDLK_UP;
                        const int d = std::clamp(depth + (up ? 1 : -1), 1, 9);
                        if (d != depth) {
                            change(scene_idx, d, use_mc, coarse, true);
                            dirty = redraw = true;
                        }
                    } else if (e.key.keysym.sym == SDLK_m && !coarse) {
                        change(scene_idx, depth, !use_mc, coarse, true);
                        dirty = redraw = true;
                    } else if (e.key.keysym.sym == SDLK_g) {
                        change(scene_idx, depth, use_mc, !coarse, true);
                        dirty = redraw = true;
                    } else if (e.key.keysym.sym == SDLK_c) { colors = !colors; redraw = true; }
                    else if (e.key.keysym.sym == SDLK_e) { edges = !edges; redraw = true; }
                    else if (e.key.keysym.sym == SDLK_v) { vertices = !vertices; redraw = true; }
                    break;
                case SDL_MOUSEBUTTONDOWN: {
                    button = e.button.button;
                    if (button == SDL_BUTTON_LEFT) {
                        const Vec3 d = cam.ray(e.button.x, e.button.y, ww, wh), oc = cam.eye - center;
                        const double b = dot(oc, d), disc = b * b - (dot(oc, oc) - 1.1 * kRadius * 1.1 * kRadius);
                        dragging_sphere = disc >= 0.0 && -b + std::sqrt(disc) > 0.0;
                        if (dragging_sphere) {
                            const Vec3 f = cam.forward();
                            const Vec3 hit = cam.eye + d * (dot(center - cam.eye, f) / dot(d, f));
                            grab_offset = center - hit;
                            animate = false;
                        }
                    }
                    break;
                }
                case SDL_MOUSEBUTTONUP: button = 0; dragging_sphere = false; break;
                case SDL_MOUSEMOTION:
                    if (button == SDL_BUTTON_LEFT && dragging_sphere) {
                        const Vec3 d = cam.ray(e.motion.x, e.motion.y, ww, wh), f = cam.forward();
                        const double den = dot(d, f);
                        if (std::fabs(den) > 1e-9) {
                            center = cam.eye + d * (dot(center - grab_offset - cam.eye, f) / den) + grab_offset;
                            dirty = true;
                        }
                    } else if (button == SDL_BUTTON_LEFT) {
                        cam.orbit(e.motion.xrel, e.motion.yrel);
                        redraw = true;
                    } else if (button == SDL_BUTTON_RIGHT) {
                        cam.dolly(std::exp(0.005 * e.motion.yrel));
                        redraw = true;
                    } else if (button == SDL_BUTTON_MIDDLE) {
                        cam.pan(e.motion.xrel, e.motion.yrel, wh);
                        redraw = true;
                    }
                    break;
                case SDL_MOUSEWHEEL:
                    cam.dolly(std::pow(0.9, e.wheel.preciseY));
                    redraw = true;
                    break;
                default: break;
                }
            }
            if (!running) break;
            if (animate) {
                // A loop around the focal point of the view.
                path_t += 2.0 * M_PI / path_frames;
                center = cam.focal + Vec3{0.5 * std::cos(path_t), 0.05 * std::sin(2.0 * path_t), 0.5 * std::sin(path_t)};
                dirty = true;
            }
            Timer tf;
            if (dirty) {
                const double c[3] = {center.x, center.y, center.z};
                Counters cnt;
                mesh = gpu->dynamic_frame_gpu(c, cnt, fs);
                dirty = false;
            }
            const double t_up = tf.seconds();
            int ww = 1, wh = 1;
            SDL_GetWindowSize(win, &ww, &wh);
            const double dist = norm(cam.eye - cam.focal);
            const Vec3 lo{base.aabb_min[0], base.aabb_min[1], base.aabb_min[2]}, hi{base.aabb_max[0], base.aabb_max[1], base.aabb_max[2]};
            const double diag = norm(hi - lo);
            const double aspect = rd.extent.height ? (double)rd.extent.width / rd.extent.height : 1.0;
            const M4 mvp = mul(perspective(kFov, aspect, std::max(0.02 * dist, 1e-3), dist + 3.0 * diag), look_at(cam.eye, cam.focal, cam.up));
            PushConstants pc{};
            memcpy(pc.mvp, mvp.m, sizeof(pc.mvp));
            pc.eye[0] = (float)cam.eye.x; pc.eye[1] = (float)cam.eye.y; pc.eye[2] = (float)cam.eye.z;
            pc.static_v = mesh.static_v;
            pc.new_v = mesh.new_v;
            pc.faces = mesh.faces;
            pc.nv_static = mesh.nv_static;
            pc.flags = colors ? 1u : 0u;
            pc.static_t = mesh.static_t;
            pc.new_t = mesh.new_t;
            Timer td;
            if (!rd.draw(pc, mesh.n_triangles, edges, vertices)) {
                vkDeviceWaitIdle(rd.dev);
                SDL_Vulkan_GetDrawableSize(win, &dw, &dh);
                rd.create_swapchain((uint32_t)dw, (uint32_t)dh);
                redraw = true;
                continue;
            }
            redraw = false;
            const double t_dr = td.seconds(), now = clock.seconds();
            frames_since_title++;
            if (bench > 0) {
                if (frame >= 3) {
                    t_frame.push_back(tf.seconds());
                    t_update.push_back(t_up);
                    t_draw.push_back(t_dr);
                    t_descent.push_back(fs.t_descent);
                    t_region.push_back(fs.t_region);
                    t_dc.push_back(fs.t_dc);
                }
                if (++frame >= bench + 3) running = false;
            }
            if (now - title_clock > 0.25) {
                char title[320];
                snprintf(title, sizeof title, "lpm_view  ·  %s, depth %d  ·  %s%s%s%s  ·  update %.1f ms  ·  %.0f fps  ·  %u triangles",
                         scene_name.c_str(), depth, mode_name(), colors ? ", colors" : "",
                         edges ? ", edges" : "", vertices ? ", vertices" : "", 1e3 * fs.total, frames_since_title / (now - title_clock),
                         mesh.n_triangles - mesh.n_empty);
                SDL_SetWindowTitle(win, title);
                title_clock = now;
                frames_since_title = 0;
            }
        }
        if (bench > 0 && !t_frame.empty()) {
            const double m = median(t_frame);
            printf("%s d%d %s (Vulkan viewer): %.1f fps (median frame %.1f ms: update %.1f = descent %.1f + region %.1f + dc %.1f; "
                   "draw and present %.1f), %u triangles\n",
                   scene_name.c_str(), depth, coarse ? "DC with coarse leaves" : use_mc ? "MC" : "DC", 1.0 / m, 1e3 * m, 1e3 * median(t_update),
                   1e3 * median(t_descent), 1e3 * median(t_region), 1e3 * median(t_dc), 1e3 * median(t_draw), mesh.n_triangles - mesh.n_empty);
        }
        rd.destroy();
    } catch (const std::exception& ex) {
        fprintf(stderr, "lpm_view: %s\n", ex.what());
        status = 1;
    }
    SDL_DestroyWindow(win);
    SDL_Quit();
    return status;
}
