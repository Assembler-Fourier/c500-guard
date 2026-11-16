"""Experimental bounded directional road motion, independent of object labels.

This measures coherent image features, not vehicles. It has no camera, network,
storage or alarm side effects. A caller must provide fresh aligned frames and
own warning/ACK policy. It is intentionally NOT enabled by importing it.
Known patterned-light counterexamples can pass raw motion: NEVER alarm from its
``hit`` alone. Require ``corroborate_motion`` and caller-owned visit/ACK acceptance.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from region_geometry import point_in_polygon, region_contains, validate_polygon


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def corroborate_motion(motion, candidates, observed_at, polygon, width, height, *, exclude_polygons=(), direction=None):
    """Return bounded weak-AI corroboration for a fresh offered motion proof.

    Candidates are full-frame pixel boxes from the ordinary model's views BEFORE
    context exclusion, each carrying its original ``observed_at`` timestamp.
    They do not enter normal AI tracking or lower its configured confidence floor.
    This is still a warning, not vehicle identity or a confirmed barrier arrival.
    """
    if (not isinstance(motion,dict) or not isinstance(candidates,(list,tuple))
            or motion.get('hit') is not True or motion.get('qualified') is not True
            or type(motion.get('pairs')) is not int or motion['pairs']<2
            or not _number(observed_at) or not _number(motion.get('observed_at'))
            or motion['observed_at']!=observed_at
            or not _number(width) or not _number(height) or width<=0 or height<=0):
        return None
    try:
        polygon=validate_polygon(polygon)
        if (not isinstance(direction,(list,tuple)) or len(direction)!=2
                or not all(_number(v) for v in direction)):return None
        axis=np.asarray(direction,dtype=float)
        if np.linalg.norm(axis)<1e-6:return None
        axis/=np.linalg.norm(axis)
        if not isinstance(exclude_polygons,(list,tuple)) or len(exclude_polygons)>16:return None
        exclusions=[validate_polygon(hole) for hole in exclude_polygons]
        raw_support=motion['support_bbox']
        if not isinstance(raw_support,(list,tuple)) or not all(_number(v) for v in raw_support):return None
        support=np.asarray(motion['support_bbox'],dtype=float)
        if (support.shape!=(4,) or not np.isfinite(support).all() or (support<0).any()
                or (support>1).any() or support[2]<=support[0] or support[3]<=support[1]):return None
        # The producer masks individual flow points; also reject a stale proof
        # centred in an area the current configuration explicitly ignores.
        centre=((support[0]+support[2])/2,(support[1]+support[3])/2)
        if any(point_in_polygon(*centre,hole) for hole in exclusions):return None
    except (KeyError,TypeError,ValueError,OverflowError):return None
    support_area=(support[2]-support[0])*(support[3]-support[1])
    edge=np.asarray(polygon,dtype=np.float32)*[width,height]
    matches=[]
    # Match the coordinator's full bounded cache. Context candidates come after
    # focused candidates and must not be starved by an old-prefix scan.
    for candidate in list(candidates)[:300]:
        if not isinstance(candidate,dict):continue
        try:
            if candidate.get('label') not in ('car','truck','bus','motorcycle'):continue
            confidence=candidate['confidence'];stamp=candidate['observed_at']
            if not _number(confidence) or not _number(stamp):continue
            raw_box=candidate['bbox']
            if not isinstance(raw_box,(list,tuple)) or not all(_number(v) for v in raw_box):continue
            box=np.asarray(candidate['bbox'],dtype=float)/[width,height,width,height]
            if (not math.isfinite(confidence) or not .30<=confidence<=1 or not math.isfinite(stamp)
                    or not 0<=observed_at-stamp<=1. or box.shape!=(4,) or not np.isfinite(box).all()
                    or (box<0).any() or (box>1).any() or box[2]<=box[0] or box[3]<=box[1]):continue
            ground=((box[0]+box[2])/2,box[3])
            if any(point_in_polygon(*ground,hole) for hole in exclusions):continue
            exact_inside=point_in_polygon(*ground,polygon)
            # One-pixel measurement tolerance at the road edge; at least one
            # member of the pair must be strictly in the configured road.
            if (not exact_inside and cv2.pointPolygonTest(edge.astype(np.float32),
                    (float(ground[0]*width),float(ground[1]*height)),True)<-1.):continue
            intersection=max(0,min(box[2],support[2])-max(box[0],support[0]))*max(
                0,min(box[3],support[3])-max(box[1],support[1]))
            coverage=intersection/support_area
            if coverage<.60:continue
            matches.append({'label':candidate['label'],'confidence':confidence,
                            'bbox_normalized':box.tolist(),'observed_at':stamp,
                            'age_s':round(observed_at-stamp,3),'motion_coverage':round(coverage,3),
                            'view':str(candidate.get('view',''))[:40], '_inside':exact_inside})
        except (TypeError,ValueError,KeyError,OverflowError):continue
    # Spatially filter before bounding association work. Newest evidence wins;
    # two views of one source frame can never supply the temporal pair.
    matches=sorted(matches,key=lambda item:(item['observed_at'],item['confidence']),reverse=True)[:60]
    for later in matches:
        b=np.asarray(later['bbox_normalized']);bs=b[2:]-b[:2]
        for earlier in matches:
            if not .2<=later['observed_at']-earlier['observed_at']<=1.:continue
            if not (later['_inside'] or earlier['_inside']):continue
            a=np.asarray(earlier['bbox_normalized']);asize=a[2:]-a[:2]
            if np.any(np.maximum(bs/asize,asize/bs)>1.6):continue
            intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            if intersection/max(1e-9,asize.prod()+bs.prod()-intersection)<.5:continue
            centre=(b[:2]+b[2:]-a[:2]-a[2:])/2
            ground=np.array([(b[0]+b[2]-a[0]-a[2])/2,b[3]-a[3]])
            coherent=True
            for travel in (centre,ground):
                forward=float(travel@axis)
                lateral=abs(float(travel[0]*axis[1]-travel[1]*axis[0]))
                if forward<.012 or np.linalg.norm(travel)>.10 or lateral>forward*.65:
                    coherent=False;break
            if not coherent:continue
            proof={key:value for key,value in later.items() if not key.startswith('_')}
            proof.update(previous_observed_at=earlier['observed_at'],
                         previous_bbox_normalized=earlier['bbox_normalized'],
                         previous_confidence=earlier['confidence'],
                         ground_progress=round(float(ground@axis),6),
                         centre_progress=round(float(centre@axis),6),candidate_observations=2)
            return proof
    return None


class ApproachMotion:
    """Two coherent forward-flow pairs with a stable exterior and a clear latch."""

    def __init__(self, polygon, direction, *, interval=.25, max_gap=1., clear_s=3., exclude_polygons=()):
        points=np.asarray(polygon,dtype=np.float32)
        vector=np.asarray(direction,dtype=np.float64)
        if (points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or len(points)>64
                or not np.isfinite(points).all() or (points<0).any() or (points>1).any()
                or vector.shape!=(2,) or not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-6):
            raise ValueError('Finite normalized polygon and nonzero direction required')
        if not (0<interval<=max_gap<=1.5 and 0<clear_s<=30):
            raise ValueError('Invalid motion timing bounds')
        if not isinstance(exclude_polygons,(list,tuple)) or len(exclude_polygons)>16:
            raise ValueError('At most 16 ignored polygons are supported')
        self.exclude_polygons=[np.asarray(validate_polygon(hole),dtype=np.float32) for hole in exclude_polygons]
        self.polygon=points;self.direction=vector
        self.interval=float(interval);self.max_gap=float(max_gap);self.clear_s=float(clear_s)
        self.reset()

    def reset(self):
        self.previous=None;self.previous_at=None;self.hits=0;self.latched=False
        self.clear_since=None;self.previous_support=None

    def _lose_pair(self):
        self.previous=None;self.previous_at=None;self.hits=0
        self.clear_since=None;self.previous_support=None

    def mark_emitted(self, observed_at):
        """Latch only after the caller accepts this exact fresh offered proof."""
        if observed_at!=self.previous_at or self.hits<2:return False
        self.latched=True
        return True

    @staticmethod
    def _flow(before,after,mask,limit):
        features=cv2.goodFeaturesToTrack(before,maxCorners=limit,qualityLevel=.025,minDistance=4,
                                        mask=mask,blockSize=5)
        if features is None:return np.empty((0,2)),np.empty((0,2))
        moved,okay,error=cv2.calcOpticalFlowPyrLK(before,after,features,None,winSize=(15,15),maxLevel=2,
                                               criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.03))
        returned,back_okay,_=cv2.calcOpticalFlowPyrLK(after,before,moved,None,winSize=(15,15),maxLevel=2,
                                                   criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.03))
        first=features.reshape(-1,2);last=moved.reshape(-1,2)
        good=(okay.ravel()!=0)&(back_okay.ravel()!=0)&(error.ravel()<20)
        good&=np.linalg.norm(first-returned.reshape(-1,2),axis=1)<=.8
        good&=np.isfinite(last).all(axis=1)
        return first[good],last[good]

    def update(self,frame,at):
        result={'hit':False,'qualified':False,'reason':'warming','pairs':0,
                'points':0,'forward_px':0.,'global_motion_px':None,'support_bbox':None,
                'observed_at':float(at) if isinstance(at,(int,float)) and math.isfinite(at) else None}
        array=np.asarray(frame)
        if (not isinstance(at,(int,float)) or not math.isfinite(at)
                or array.dtype!=np.uint8 or array.ndim not in (2,3)
                or (array.ndim==3 and array.shape[2]!=3) or min(array.shape[:2])<32):
            self._lose_pair();result['reason']='invalid frame';return result
        gray=cv2.cvtColor(array,cv2.COLOR_RGB2GRAY) if array.ndim==3 else array
        # Bound work and memory independently of source/display resolution.
        scale=min(1.,640./gray.shape[1],360./gray.shape[0])
        if scale<1:gray=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        gray=np.ascontiguousarray(gray)
        if self.previous_at is not None:
            elapsed=at-self.previous_at
            if 0<=elapsed<self.interval:
                result['reason']='waiting for fresh interval';return result
            if elapsed<0 or elapsed>self.max_gap or gray.shape!=self.previous.shape:
                self._lose_pair();result['reason']='frame gap'
        before=self.previous
        self.previous=gray.copy();self.previous_at=float(at)
        if before is None:return result
        h,w=gray.shape;region=np.zeros(gray.shape,np.uint8)
        cv2.fillPoly(region,[np.round(self.polygon*(w,h)).astype(np.int32)],255)
        region=cv2.erode(region,np.ones((3,3),np.uint8))
        exterior=cv2.bitwise_not(cv2.dilate(region,np.ones((31,31),np.uint8)))
        exterior[:12]=0;exterior[-6:]=0;exterior[:,:6]=0;exterior[:,-6:]=0
        if self.exclude_polygons:
            ignored=np.zeros(gray.shape,np.uint8)
            cv2.fillPoly(ignored,[np.round(hole*(w,h)).astype(np.int32) for hole in self.exclude_polygons],255)
            region[cv2.dilate(ignored,np.ones((3,3),np.uint8))!=0]=0
        p0,p1=self._flow(before,gray,exterior,160)
        if len(p0)<20:
            return self._finish(result,at,'insufficient static context')
        flow=p1-p0;global_flow=np.median(flow,axis=0)
        global_motion=float(np.linalg.norm(global_flow))
        result['global_motion_px']=round(global_motion,3)
        if global_motion>.8 or np.mean(np.linalg.norm(flow,axis=1)>1.5)>.35:
            return self._finish(result,at,'camera or widespread motion')
        offset=float(np.median(gray[exterior!=0].astype(float)-before[exterior!=0].astype(float)))
        if abs(offset)>10:
            return self._finish(result,at,'global exposure change')
        mask=region.copy();mask[(before<20)|(before>240)]=0
        # Track local structure rather than an illumination gradient. A moving
        # headlight/shadow wash can make raw-luma LK follow light across fixed
        # road texture; removing low spatial frequencies keeps that texture put.
        def structure(image):
            floating=image.astype(np.float32)
            detail=floating-cv2.GaussianBlur(floating,(0,0),4)
            return np.clip(128+2*detail,0,255).astype(np.uint8)
        p0,p1=self._flow(structure(before),structure(gray),mask,140)
        result['tracked_points']=len(p0)
        if len(p0)<6:return self._finish(result,at,'insufficient road texture')
        travel=p1-p0-global_flow
        d=self.direction*np.array([w,h]);d=d/np.linalg.norm(d)
        forward=travel@d;side=np.abs(travel[:,0]*d[1]-travel[:,1]*d[0])
        inside=(p1[:,0]>=0)&(p1[:,0]<w)&(p1[:,1]>=0)&(p1[:,1]<h)
        indices=np.clip(np.round(p1).astype(int),[0,0],[w-1,h-1])
        inside&=region[indices[:,1],indices[:,0]]!=0
        selected=inside&(forward>=1.8)&(forward<=45)&(side<=forward*.65)
        if np.count_nonzero(selected)<6:return self._finish(result,at,'no coherent forward movement')
        pts=p1[selected];vectors=travel[selected];projection=forward[selected]
        # Only one compact group may qualify; scattered moving foliage does not.
        distance=np.linalg.norm(pts[:,None,:]-pts[None,:,:],axis=2)
        groups=[];unseen=set(range(len(pts)))
        while unseen:
            stack=[unseen.pop()];group=[]
            while stack:
                index=stack.pop();group.append(index)
                neighbours=unseen.intersection(np.flatnonzero(distance[index]<=32).tolist())
                unseen.difference_update(neighbours);stack.extend(neighbours)
            groups.append(group)
        group=max(groups,key=len);pts=pts[group];vectors=vectors[group];projection=projection[group]
        result['points']=len(pts)
        if len(pts)<6:return self._finish(result,at,'scattered motion')
        median=np.median(vectors,axis=0);coherence=np.linalg.norm(vectors-median,axis=1)
        if np.mean(coherence<=max(2.,np.linalg.norm(median)*.5))<.75:
            return self._finish(result,at,'inconsistent local motion')
        low=pts.min(axis=0);high=pts.max(axis=0)
        if high[0]-low[0]<10 or high[1]-low[1]<5:
            return self._finish(result,at,'thin edge motion')
        result['forward_px']=round(float(np.median(projection)),3)
        result['support_bbox']=[float(low[0]/w),float(low[1]/h),float(high[0]/w),float(high[1]/h)]
        result['qualified']=True
        return self._finish(result,at,'forward road movement')

    def _finish(self,result,at,reason):
        result['reason']=reason
        if result['qualified']:
            support=np.array(result['support_bbox']);centre=(support[:2]+support[2:])/2
            if self.previous_support is not None and np.linalg.norm(centre-self.previous_support)>.15:self.hits=0
            self.previous_support=centre;self.hits+=1;self.clear_since=None
            if self.hits>=2 and not self.latched:
                result['hit']=True
        else:
            self.hits=0;self.previous_support=None
            # Only fresh, valid quiet pairs prove a clear episode. Global-camera
            # uncertainty and exposure transitions cannot rearm an old warning.
            if reason in ('no coherent forward movement','insufficient road texture'):
                self.clear_since=at if self.clear_since is None else self.clear_since
                if at-self.clear_since>=self.clear_s:self.latched=False
            else:self.clear_since=None
        result['pairs']=self.hits
        return result
